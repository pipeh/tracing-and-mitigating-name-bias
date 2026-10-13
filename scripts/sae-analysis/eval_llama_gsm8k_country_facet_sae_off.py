#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate GSM8K with baseline and single-layer SAE-off conditions.

Feature CSVs contain country, topic, and salient_features. This module also
provides model loading and generation helpers for the multilayer benchmarks."""

import argparse
import ast
import gc
import json
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_LAYER = 12
DEFAULT_SITE = "R"
DEFAULT_EXPANSION = "8x"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_OUTPUT_DIR = "results_gsm8k_country_facet_sae_off"
DEFAULT_MAX_INPUT_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 256


# Paths / imports for SAE


def add_sae_paths(repo_root: Optional[str] = None) -> Path:
    """Find ``scripts/sae-analysis/utils.py`` and add it to ``sys.path``."""
    candidates: List[Path] = []

    if repo_root is not None:
        candidates.append(Path(repo_root).resolve())

    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    candidates.extend([cwd, *cwd.parents, script_dir, *script_dir.parents])

    seen = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)

        sae_dir = root / "scripts" / "sae-analysis"
        utils_file = sae_dir / "utils.py"

        if utils_file.exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            if str(sae_dir) not in sys.path:
                sys.path.insert(0, str(sae_dir))
            return root

    raise FileNotFoundError(
        "No pude encontrar scripts/sae-analysis/utils.py. "
        "Ejecuta el script desde la raíz del repositorio o usa "
        "--repo_root /path/to/repository."
    )


def slugify(text: str) -> str:
    """Create a filesystem-safe, readable identifier."""
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_") or "unknown"


def get_torch_dtype(dtype: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


# GSM8K answer extraction


def normalize_number(x):
    if x is None:
        return None

    x = str(x)
    x = x.replace(",", "")
    x = x.replace("$", "")
    x = x.replace("%", "")
    x = x.strip()

    match = re.search(r"-?\d+(?:\.\d+)?", x)
    if not match:
        return None

    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def extract_gold_answer(answer_text: str):
    """GSM8K stores the final answer after ``####``."""
    if "####" in str(answer_text):
        answer = str(answer_text).split("####")[-1].strip()
    else:
        answer = str(answer_text).strip()
    return normalize_number(answer)


def extract_pred_answer(text: str):
    """Extract the final number, prioritizing content after ``####``."""
    text = str(text)
    candidate = text.split("####")[-1] if "####" in text else text

    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", candidate)
    if not numbers:
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not numbers:
        return None

    return normalize_number(numbers[-1])


# Prompt


def build_zero_shot_prompt(question: str) -> str:
    return f"""Solve the following grade-school math problem step by step.

At the end, write the final numerical answer in this exact format:
#### <number>

Question:
{question}

Answer:"""


def apply_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful mathematical reasoning assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# Country/facet feature file


def parse_feature_ids_cell(cell) -> List[int]:
    """
    Parse feature IDs from Python lists, comma-separated strings, or
    NumPy-style strings such as ``[123 456 789]``.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    if isinstance(cell, (list, tuple, set)):
        return [int(x) for x in cell if x is not None]

    if isinstance(cell, int):
        return [int(cell)]

    raw = str(cell).strip()
    if not raw:
        return []

    # First try a normal Python literal, e.g. [123, 456].
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, int):
            return [int(parsed)]
        if isinstance(parsed, (list, tuple, set)):
            return [int(x) for x in parsed if x is not None]
    except (ValueError, SyntaxError):
        pass

    # Robust fallback for NumPy-style arrays without commas/newline-separated.
    values = re.findall(r"-?\d+", raw)
    return [int(x) for x in values]


def load_country_facet_groups(
    path: Path,
    country_col: str,
    facet_col: str,
    feature_col: str,
    top_k_features: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pd.read_pickle(f)
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    else:
        df = pd.read_csv(path)

    required = {country_col, facet_col, feature_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    groups: List[Dict] = []
    seen_pairs = set()

    for row_idx, row in df.iterrows():
        country = str(row[country_col]).strip()
        facet = str(row[facet_col]).strip().lower()
        pair = (country, facet)

        if pair in seen_pairs:
            raise ValueError(
                f"The country={country!r}, facet={facet!r} "
                "occurs more than once in the CSV."
            )
        seen_pairs.add(pair)

        feature_ids = parse_feature_ids_cell(row[feature_col])
        # Preserve the original ranking while removing duplicates.
        feature_ids = list(dict.fromkeys(feature_ids))

        if top_k_features is not None:
            feature_ids = feature_ids[:top_k_features]

        if not feature_ids:
            raise ValueError(
                f"Could not read features in row {row_idx}: "
                f"country={country!r}, facet={facet!r}, value={row[feature_col]!r}"
            )

        groups.append(
            {
                "row_index": int(row_idx),
                "country": country,
                "facet": facet,
                "country_slug": slugify(country),
                "facet_slug": slugify(facet),
                "feature_ids": feature_ids,
                "n_features": len(feature_ids),
            }
        )

    groups.sort(key=lambda x: (x["country"].lower(), x["facet"].lower()))
    return df, groups


def build_sae_disable_hook(sae, disabled_feature_ids: Sequence[int]):
    """
    Remove the selected SAE decoder contribution from each hidden state:

        selected_features = activation(encoder_selected(hidden_states))
        delta = selected_features @ decoder_selected.T
        new_hidden_states = hidden_states - delta
    """
    if not disabled_feature_ids:
        return None

    device = next(sae.parameters()).device
    dtype = next(sae.parameters()).dtype

    disabled_ids = torch.tensor(
        sorted(set(int(fid) for fid in disabled_feature_ids)),
        device=device,
        dtype=torch.long,
    )

    d_sae = sae.enc.weight.shape[0]
    for fid in disabled_feature_ids:
        if fid < 0 or fid >= d_sae:
            raise ValueError(f"Feature id {fid} outside the range [0, {d_sae}).")

    enc_w = sae.enc.weight.index_select(0, disabled_ids).to(
        device=device, dtype=dtype
    )

    enc_b = None
    if getattr(sae.enc, "bias", None) is not None:
        enc_b = sae.enc.bias.index_select(0, disabled_ids).to(
            device=device, dtype=dtype
        )

    if not isinstance(sae.dec, torch.nn.Linear):
        raise TypeError("The selective hook requires sae.dec to be torch.nn.Linear.")

    dec_w = sae.dec.weight.index_select(1, disabled_ids).to(
        device=device, dtype=dtype
    )

    def hook_fn(module, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        if hs is None:
            return out

        selected_feats = F.linear(hs, enc_w, enc_b)
        selected_feats = sae.act(selected_feats)

        delta = selected_feats @ dec_w.T
        new_hs = hs - delta

        if isinstance(out, tuple):
            return (new_hs, *out[1:])
        if isinstance(out, list):
            return [new_hs, *out[1:]]
        return new_hs

    return hook_fn


def load_model_and_tokenizer(
    model_id: str,
    dtype: str,
    trust_remote_code: bool,
    load_in_8bit: bool,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = get_torch_dtype(dtype)

    if load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model


def get_layer_device(model, layer: int):
    return next(model.model.layers[layer].parameters()).device


def load_sae_for_layer(layer: int, site: str, expansion: str, model, dtype: str):
    from utils import load_llamascope_sae

    layer_device = get_layer_device(model, layer)
    torch_dtype = get_torch_dtype(dtype)

    sae, hp, _ = load_llamascope_sae(
        layer=layer,
        site=site,
        expansion=expansion,
        device=layer_device,
        dtype=torch_dtype,
    )

    sae.eval()
    return sae, hp


# Generation


@torch.no_grad()
def generate_one(
    tokenizer,
    model,
    prompt: str,
    max_input_length: int,
    max_new_tokens: int,
    *,
    serialized_chat: bool = False,
):
    input_device = model.get_input_embeddings().weight.device
    if serialized_chat:
        from gsm8k_meta_prompt import tokenize_meta_prompt

        enc = tokenize_meta_prompt(tokenizer, prompt, max_input_length)
    else:
        input_text = apply_chat_template(tokenizer, prompt)
        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
    enc = {k: v.to(input_device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[-1]

    terminators = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        terminators.append(eot_id)

    generated = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
    )

    output_ids = generated[0][input_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# Evaluation and output paths


def condition_paths(
    args,
    condition: str,
    country: Optional[str],
    facet: Optional[str],
) -> Tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = slugify(args.model_id)
    architecture_tag = f"L{args.layer}{args.site}_{args.expansion}"

    if condition == "baseline":
        condition_tag = "baseline"
    else:
        condition_tag = f"sae_disabled_{slugify(country)}_{slugify(facet)}"

    sample_tag = f"N{args.n_samples}" if args.n_samples is not None else "Nall"
    stem = (
        f"gsm8k_{args.split}_{sample_tag}_{condition_tag}_"
        f"{model_tag}_{architecture_tag}"
    )
    return output_dir / f"{stem}.csv", output_dir / f"metrics_{stem}.json"


def evaluate_condition(
    condition: str,
    tokenizer,
    model,
    sae,
    disabled_feature_ids: List[int],
    args,
    test_data,
    country: Optional[str] = None,
    facet: Optional[str] = None,
):
    output_path, metrics_path = condition_paths(args, condition, country, facet)

    if args.skip_existing and output_path.exists() and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[skip] Ya existe: {metrics_path}")
        return None, metrics

    hook_handle = None
    if disabled_feature_ids:
        hook_fn = build_sae_disable_hook(sae, disabled_feature_ids)
        hook_handle = model.model.layers[args.layer].register_forward_hook(hook_fn)
        print(
            f"[info] Hook: layer={args.layer}, site={args.site}, "
            f"expansion={args.expansion}"
        )
        print(
            f"[info] Intervención: country={country}, facet={facet}, "
            f"n_features={len(disabled_feature_ids)}"
        )
        print(f"[info] Disabled features: {disabled_feature_ids}")

    rows = []
    correct = 0
    description = "baseline" if condition == "baseline" else f"{country} | {facet}"

    try:
        for i, row in enumerate(tqdm(test_data, desc=description)):
            question = row["question"]
            gold_full = row["answer"]
            gold_answer = extract_gold_answer(gold_full)
            prompt = build_zero_shot_prompt(question)

            prediction_text = generate_one(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
            prediction_answer = extract_pred_answer(prediction_text)

            is_correct = int(
                prediction_answer is not None
                and gold_answer is not None
                and prediction_answer == gold_answer
            )
            correct += is_correct

            rows.append(
                {
                    "id": i,
                    "condition": condition,
                    "country": country,
                    "facet": facet,
                    "model_id": args.model_id,
                    "layer": args.layer,
                    "site": args.site,
                    "expansion": args.expansion,
                    "n_disabled_features": len(disabled_feature_ids),
                    "disabled_features": json.dumps(disabled_feature_ids),
                    "question": question,
                    "gold_full": gold_full,
                    "gold_answer": (
                        str(gold_answer) if gold_answer is not None else None
                    ),
                    "prediction_text": prediction_text,
                    "prediction_answer": (
                        str(prediction_answer)
                        if prediction_answer is not None
                        else None
                    ),
                    "correct": is_correct,
                }
            )

            if (i + 1) % args.save_every == 0:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    finally:
        if hook_handle is not None:
            hook_handle.remove()
            print("[info] Hook eliminado")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    total = len(test_data)
    accuracy = correct / total if total else float("nan")

    metrics = {
        "model_id": args.model_id,
        "dataset": "openai/gsm8k",
        "split": args.split,
        "setting": "zero-shot",
        "condition": condition,
        "country": country,
        "facet": facet,
        "n_samples": total,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "layer": args.layer,
        "site": args.site,
        "expansion": args.expansion,
        "n_disabled_features": len(disabled_feature_ids),
        "disabled_features": disabled_feature_ids,
        "predictions_file": str(output_path),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nResults")
    print(f"Condition: {condition}")
    print(f"Country: {country}")
    print(f"Facet: {facet}")
    print(f"Disabled features: {len(disabled_feature_ids)}")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"CSV: {output_path}")
    print(f"JSON: {metrics_path}")

    return df, metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_file", type=Path, required=True)
    parser.add_argument("--country_col", type=str, default="country")
    parser.add_argument("--facet_col", type=str, default="topic")
    parser.add_argument("--feature_col", type=str, default="salient_features")
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=None,
        help="Use only the first K ranked features in each CSV row.",
    )

    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--site", type=str, choices=["R", "A", "M", "TC"], default=DEFAULT_SITE
    )
    parser.add_argument(
        "--expansion", type=str, choices=["8x", "32x"], default=DEFAULT_EXPANSION
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default=DEFAULT_DTYPE,
    )
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument(
        "--no_trust_remote_code",
        dest="trust_remote_code",
        action="store_false",
    )

    parser.add_argument("--max_input_length", type=int, default=DEFAULT_MAX_INPUT_LENGTH)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--save_every", type=int, default=20)

    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Run only the country-facet interventions.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Do not rerun conditions whose prediction CSV and metric JSON exist.",
    )
    parser.add_argument("--repo_root", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.top_k_features is not None and args.top_k_features <= 0:
        raise ValueError("--top_k_features must be positive.")

    add_sae_paths(args.repo_root)

    source_df, groups = load_country_facet_groups(
        path=args.features_file,
        country_col=args.country_col,
        facet_col=args.facet_col,
        feature_col=args.feature_col,
        top_k_features=args.top_k_features,
    )

    print(f"[info] CSV rows: {len(source_df)}")
    print(f"[info] Country-facet combinations: {len(groups)}")
    print(
        "[info] Countries: "
        + ", ".join(sorted({group["country"] for group in groups}))
    )
    print(
        "[info] Facets: "
        + ", ".join(sorted({group["facet"] for group in groups}))
    )

    parsed_groups_path = Path(args.output_dir) / "parsed_country_facet_features.csv"
    parsed_groups_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "country": group["country"],
                "facet": group["facet"],
                "n_features": group["n_features"],
                "feature_ids": json.dumps(group["feature_ids"]),
            }
            for group in groups
        ]
    ).to_csv(parsed_groups_path, index=False)
    print(f"[info] Parsed features: {parsed_groups_path}")

    print("[info] Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset[args.split]

    if args.n_samples is not None:
        test_data = test_data.select(range(min(args.n_samples, len(test_data))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")
    print(f"[info] Model: {args.model_id}")

    tokenizer, model = load_model_and_tokenizer(
        model_id=args.model_id,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        load_in_8bit=args.load_in_8bit,
        device=device,
    )

    print("[info] Loading LlamaScope SAE...")
    sae, hp = load_sae_for_layer(
        layer=args.layer,
        site=args.site,
        expansion=args.expansion,
        model=model,
        dtype=args.dtype,
    )

    all_metrics: List[Dict] = []

    if not args.skip_baseline:
        _, baseline_metrics = evaluate_condition(
            condition="baseline",
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            disabled_feature_ids=[],
            args=args,
            test_data=test_data,
            country=None,
            facet=None,
        )
        all_metrics.append(baseline_metrics)

    for position, group in enumerate(groups, start=1):
        print("\n" + "=" * 80)
        print(
            f"[info] Combination {position}/{len(groups)}: "
            f"{group['country']} × {group['facet']}"
        )

        _, metrics = evaluate_condition(
            condition="sae_disabled",
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            disabled_feature_ids=group["feature_ids"],
            args=args,
            test_data=test_data,
            country=group["country"],
            facet=group["facet"],
        )
        all_metrics.append(metrics)

        # Update summaries after every condition, so partial progress is retained.
        summary_json_path = Path(args.output_dir) / "summary_metrics.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        summary_df = pd.DataFrame(all_metrics)
        baseline_rows = summary_df[summary_df["condition"] == "baseline"]
        if not baseline_rows.empty:
            baseline_accuracy = float(baseline_rows.iloc[0]["accuracy"])
            summary_df["baseline_accuracy"] = baseline_accuracy
            summary_df["accuracy_delta_vs_baseline"] = (
                summary_df["accuracy"] - baseline_accuracy
            )
            summary_df["accuracy_drop_vs_baseline"] = (
                baseline_accuracy - summary_df["accuracy"]
            )
        summary_df.to_csv(Path(args.output_dir) / "summary_metrics.csv", index=False)

    # In case there were no intervention groups, still save the final summary.
    summary_json_path = Path(args.output_dir) / "summary_metrics.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(all_metrics)
    if not summary_df.empty:
        baseline_rows = summary_df[summary_df["condition"] == "baseline"]
        if not baseline_rows.empty:
            baseline_accuracy = float(baseline_rows.iloc[0]["accuracy"])
            summary_df["baseline_accuracy"] = baseline_accuracy
            summary_df["accuracy_delta_vs_baseline"] = (
                summary_df["accuracy"] - baseline_accuracy
            )
            summary_df["accuracy_drop_vs_baseline"] = (
                baseline_accuracy - summary_df["accuracy"]
            )
        summary_df.to_csv(Path(args.output_dir) / "summary_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print(f"[done] Summary JSON: {summary_json_path}")
    print(f"[done] Summary CSV: {Path(args.output_dir) / 'summary_metrics.csv'}")

    del model
    del tokenizer
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
