#!/usr/bin/env python
"""Evaluate GSM8K with baseline and seven-layer SAE-off conditions.

Defaults: Meta's 8-shot prompt, greedy generation, and 1024 output tokens.
Scoring requires the final-answer marker and numeric exact match. Each
country/facet intervention disables 20 features per layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path

from dotenv import dotenv_values
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "scripts" / "experiment"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))
from generation_utils import parse_feature_ids  # noqa: E402
from gsm8k_meta_prompt import build_meta_prompt, extract_meta_answer, load_meta_prompt  # noqa: E402
from utils import JumpReLU

# Reuse the original GSM8K prompt, answer extraction, model loading, and
# generation code so results remain directly comparable to that evaluator.
BASE_PATH = Path(__file__).with_name("eval_llama_gsm8k_country_facet_sae_off.py")
def load_base():
    spec = importlib.util.spec_from_file_location("gsm8k_country_facet_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the original GSM8K evaluator: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

LAYERS = (4, 8, 12, 16, 20, 24, 28)
ENV = dotenv_values(ROOT / ".env")
DEFAULT_FEATURES_DIR = Path(ENV.get("DATA_DIR", "data")) / "crisp" / ENV.get("JUDGE_MODEL_ID", "llama-3.1-70b-w8a8")
DEFAULT_MODEL = ENV.get("GENERATION_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
def default_output_dir() -> Path:
    data_dir = dotenv_values(ROOT / ".env").get("DATA_DIR")
    if not data_dir:
        raise ValueError(f"DATA_DIR is missing from {ROOT / '.env'}; provide --output_dir")
    data_path = Path(data_dir).expanduser()
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    return data_path / "gsm8k"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--feature_model_name", default=ENV.get("GENERATION_MODEL_SHORT", "Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--country", action="append", help="Filter country; repeat for several")
    parser.add_argument("--facet", action="append", help="Filter topic/facet; repeat for several")
    parser.add_argument("--num_features_per_layer", type=int, default=20)
    parser.add_argument("--model_id", default=str(DEFAULT_MODEL))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prompt_protocol", choices=("meta", "train"), default="meta", help="meta: published Llama 3.1 8B Instruct 8-shot prompt; train: previous prompt with the first N training examples")
    parser.add_argument("--shots", type=int, help="meta requires 8 (default); train defaults to 0")
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--output_dir", type=Path, help="Output directory (default: DATA_DIR/gsm8k from .env)")
    parser.add_argument("--expansion", choices=("8x", "32x"), default="8x")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="float16")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_input_length", type=int, help="Default: 3072 for meta, 4096 for train; meta prompts are never truncated")
    parser.add_argument("--max_new_tokens", type=int, help="Default: 1024 for meta, 256 for train")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--resume_partial", action="store_true", help="Continue a condition from its checkpoint CSV after validating its rows")
    parser.add_argument("--list_conditions", action="store_true", help="Validate and list feature groups without loading GSM8K or the model")
    args = parser.parse_args()
    if args.shots is None:
        args.shots = 8 if args.prompt_protocol == "meta" else 0
    if args.prompt_protocol == "meta" and args.shots != 8:
        parser.error("--prompt_protocol meta requires --shots 8; use --prompt_protocol train for other shot counts")
    if args.prompt_protocol == "meta" and args.split != "test":
        parser.error("--prompt_protocol meta requires --split test")
    if args.max_input_length is None:
        args.max_input_length = 3072 if args.prompt_protocol == "meta" else 4096
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.prompt_protocol == "meta" else 256
    args.meta_prompt = load_meta_prompt() if args.prompt_protocol == "meta" else None
    args.evaluation_config = {
        "prompt_protocol": args.prompt_protocol, "shots": args.shots,
        "template_sha256": args.meta_prompt["template_sha256"] if args.meta_prompt else None,
        "max_input_length": args.max_input_length, "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype, "load_in_8bit": args.load_in_8bit,
        "do_sample": False,
        "answer_parser": "meta_final_marker_numeric_v1" if args.meta_prompt else "legacy_last_number_v1",
    }
    args.evaluation_id = hashlib.sha256(json.dumps(args.evaluation_config, sort_keys=True).encode()).hexdigest()[:12]
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    for name in ("num_features_per_layer", "max_input_length", "max_new_tokens", "save_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.n_samples is not None and args.n_samples <= 0:
        parser.error("--n_samples must be positive")
    if args.shots < 0:
        parser.error("--shots cannot be negative")
    if args.shots and args.split != "test":
        parser.error("Few-shot benchmarking requires --split test, keeping demonstrations separate from evaluated questions")
    return args


def feature_path(args, layer):
    return args.features_dir / (
        f"topic_country_salient_features_{args.feature_model_name}_L{layer}R_{args.expansion}.parquet"
    )


def load_groups(args):
    groups = {}
    expected_pairs = None
    for layer in LAYERS:
        path = feature_path(args, layer)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature Parquet for L{layer}R: {path}")
        frame = pd.read_parquet(path, columns=["country", "topic", "salient_features"])
        pairs = set()
        for row in frame.itertuples(index=False):
            country, facet = str(row.country).strip(), str(row.topic).strip().lower()
            key = (country, facet)
            if key in pairs:
                raise ValueError(f"Duplicate row {key} in {path}")
            pairs.add(key)
            features = parse_feature_ids(row.salient_features)
            selected = features[: args.num_features_per_layer]
            if len(selected) != args.num_features_per_layer:
                raise ValueError(f"Expected {args.num_features_per_layer} features for {key} at L{layer}R in {path}; got {len(selected)}")
            groups.setdefault(key, {})[layer] = selected
        if expected_pairs is not None and pairs != expected_pairs:
            raise ValueError(f"Country/facet rows in {path} differ from the first layer: missing={sorted(expected_pairs - pairs)}, extra={sorted(pairs - expected_pairs)}")
        expected_pairs = pairs
    countries = {x.casefold() for x in args.country or []}
    facets = {x.casefold() for x in args.facet or []}
    available_countries = {country.casefold() for country, _ in groups}
    missing_countries = countries - available_countries
    if missing_countries:
        raise ValueError(
            f"Countries missing from feature Parquets: {sorted(missing_countries)}. "
            f"Available: {sorted({country for country, _ in groups})}"
        )
    selected_groups = {
        key: groups[key] for key in sorted(groups)
        if (not countries or key[0].casefold() in countries)
        and (not facets or key[1].casefold() in facets)
    }
    if not selected_groups:
        raise ValueError(f"No country/facet rows matched filters. Available: {sorted(groups)}")
    return selected_groups


def selected_tensors(sae, layer, features):
    """Retain only selected SAE weights, including JumpReLU thresholds."""
    if not isinstance(sae.enc, torch.nn.Linear) or not isinstance(sae.dec, torch.nn.Linear):
        raise TypeError("Expected linear SAE encoder and decoder")
    width = sae.enc.weight.shape[0]
    if any(fid < 0 or fid >= width for fid in features):
        raise ValueError(f"Invalid L{layer}R feature ID; valid range is [0, {width})")
    idx = torch.tensor(features, dtype=torch.long, device=sae.enc.weight.device)
    weights = sae.enc.weight.index_select(0, idx).detach().clone()
    bias = sae.enc.bias.index_select(0, idx).detach().clone() if sae.enc.bias is not None else None
    decoder = sae.dec.weight.index_select(1, idx).detach().clone()
    activation = sae.act
    activation_kind = "jumprelu"
    if hasattr(activation, "get_jumprelu_threshold"):
        thresholds = activation.get_jumprelu_threshold().reshape(-1).index_select(0, idx).detach().clone()
    elif hasattr(activation, "log_jumprelu_threshold"):
        thresholds = activation.log_jumprelu_threshold.exp().reshape(-1).index_select(0, idx).detach().clone()
    elif isinstance(activation, JumpReLU):
        thresholds = torch.tensor(activation.threshold, device=weights.device, dtype=weights.dtype)
        activation_kind = "shifted_relu"
    elif "relu" in type(activation).__name__.lower():
        thresholds = None
        activation_kind = "relu"
    else:
        raise TypeError(f"Unsupported SAE activation: {type(activation).__name__}")
    return weights, bias, decoder, thresholds, activation_kind


def make_hook(tensors):
    weights, bias, decoder, thresholds, activation_kind = tensors

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        pre = F.linear(hidden, weights, bias)
        if activation_kind == "jumprelu":
            active = pre * (pre > thresholds)
        elif activation_kind == "shifted_relu":
            active = F.relu(pre - thresholds)
        else:
            active = F.relu(pre)
        modified = hidden - active @ decoder.T
        if isinstance(output, tuple):
            return (modified, *output[1:])
        if isinstance(output, list):
            return [modified, *output[1:]]
        return modified

    return hook


def load_hooks(model, layer_features, expansion, dtype):
    from utils import load_llamascope_sae

    hooks = []
    for layer, features in layer_features.items():
        model_layer = model.model.layers[layer]
        device = next(model_layer.parameters()).device
        sae, _, _ = load_llamascope_sae(
            layer=layer, site="R", expansion=expansion,
            device=device, dtype=base.get_torch_dtype(dtype),
        )
        hooks.append((layer, make_hook(selected_tensors(sae, layer, features))))
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return hooks


def output_paths(args, country=None, facet=None):
    sample = f"N{args.n_samples}" if args.n_samples is not None else "Nall"
    condition = "baseline" if country is None else f"sae_disabled_{base.slugify(country)}_{base.slugify(facet)}"
    setting_tag = f"_{args.prompt_protocol}_{args.shots}shot_{args.evaluation_id}"
    stem = f"gsm8k_{args.split}_{sample}{setting_tag}_{condition}_{base.slugify(args.model_id)}_L4R-L8R-L12R-L16R-L20R-L24R-L28R_{args.expansion}_top{args.num_features_per_layer}"
    return args.output_dir / f"{stem}.csv", args.output_dir / f"metrics_{stem}.json"


def summary_paths(args):
    suffix = f"_{args.prompt_protocol}_{args.shots}shot_{args.evaluation_id}_{base.slugify(args.model_id)}_{args.split}_N{args.n_samples or 'all'}_{args.expansion}_top{args.num_features_per_layer}"
    return (
        args.output_dir / f"summary_metrics{suffix}.json",
        args.output_dir / f"summary_metrics{suffix}.csv",
    )


def build_few_shot_prompt(question, demonstrations):
    parts = [
        "Solve the following grade-school math problem step by step.",
        "At the end, write the final numerical answer in this exact format:",
        "#### <number>",
    ]
    for example in demonstrations:
        parts.extend(("", f"Question:\n{example['question']}", f"Answer:\n{example['answer']}"))
    parts.extend(("", f"Question:\n{question}", "Answer:"))
    return "\n".join(parts)


def load_partial_rows(path, args, data, country, facet, layer_features):
    if not path.exists():
        return []
    frame = pd.read_csv(path, keep_default_na=False)
    expected_features = {f"L{k}R": v for k, v in (layer_features or {}).items()}
    required = {"id", "condition", "country", "facet", "model_id", "disabled_features_by_layer", "question", "gold_full", "prediction_text", "correct", "evaluation_id"}
    if args.shots:
        required.add("shots")
    if not required.issubset(frame.columns):
        raise ValueError(f"Cannot resume {path}: missing columns {sorted(required - set(frame.columns))}")
    if len(frame) > len(data):
        raise ValueError(f"Cannot resume {path}: {len(frame)} rows exceed {len(data)} questions")
    for i, row in enumerate(frame.itertuples(index=False)):
        if (
            int(row.id) != i
            or row.condition != ("sae_disabled" if country else "baseline")
            or row.country != (country or "")
            or row.facet != (facet or "")
            or row.model_id != args.model_id
            or row.evaluation_id != args.evaluation_id
            or (args.shots and int(row.shots) != args.shots)
            or json.loads(row.disabled_features_by_layer) != expected_features
            or row.question != data[i]["question"]
            or row.gold_full != data[i]["answer"]
            or row.prediction_text == ""
            or int(row.correct) not in (0, 1)
        ):
            raise ValueError(f"Cannot resume {path}: checkpoint row {i} does not match this evaluation")
    print(f"[resume] {path}: {len(frame)}/{len(data)} completed questions")
    return frame.to_dict("records")


def evaluate(args, tokenizer, model, data, demonstrations, country=None, facet=None, layer_features=None):
    prediction_path, metrics_path = output_paths(args, country, facet)
    if args.skip_existing and prediction_path.exists() and metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_features = {f"L{k}R": v for k, v in (layer_features or {}).items()}
        if existing.get("evaluation_config") != args.evaluation_config or existing.get("disabled_features_by_layer") != expected_features or existing.get("n_samples") != len(data):
            raise ValueError(f"Existing metrics do not match this evaluation: {metrics_path}")
        print(f"[skip] {metrics_path}")
        return existing
    rows = load_partial_rows(prediction_path, args, data, country, facet, layer_features) if args.resume_partial else []
    correct = sum(int(row["correct"]) for row in rows)
    start = len(rows)
    hooks = load_hooks(model, layer_features, args.expansion, args.dtype) if layer_features and start < len(data) else []
    with ExitStack() as stack:
        for layer, hook in hooks:
            handle = model.model.layers[layer].register_forward_hook(hook)
            stack.callback(handle.remove)
        for i in tqdm(range(start, len(data)), desc=f"{country}/{facet}" if country else "baseline"):
            example = data[i]
            gold = base.extract_gold_answer(example["answer"])
            prompt = (
                build_meta_prompt(example["question"], args.meta_prompt)
                if args.meta_prompt else (
                    build_few_shot_prompt(example["question"], demonstrations)
                    if args.shots else base.build_zero_shot_prompt(example["question"])
                )
            )
            text = base.generate_one(
                tokenizer, model, prompt,
                args.max_input_length, args.max_new_tokens,
                serialized_chat=args.meta_prompt is not None,
            )
            predicted = extract_meta_answer(text) if args.meta_prompt else base.extract_pred_answer(text)
            is_correct = int(predicted is not None and gold is not None and predicted == gold)
            correct += is_correct
            rows.append({
                "id": i, "condition": "sae_disabled" if country else "baseline",
                "country": country, "facet": facet, "model_id": args.model_id,
                "shots": args.shots,
                "evaluation_id": args.evaluation_id,
                "disabled_features_by_layer": json.dumps({f"L{k}R": v for k, v in (layer_features or {}).items()}),
                "n_disabled_features": sum(map(len, (layer_features or {}).values())),
                "question": example["question"], "gold_full": example["answer"],
                "gold_answer": str(gold) if gold is not None else None,
                "prediction_text": text,
                "prediction_answer": str(predicted) if predicted is not None else None,
                "correct": is_correct,
            })
            if (i + 1) % args.save_every == 0:
                pd.DataFrame(rows).to_csv(prediction_path, index=False)
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    metrics = {
        "model_id": args.model_id, "dataset": "openai/gsm8k", "split": args.split,
        "setting": f"{args.shots}-shot" if args.shots else "zero-shot",
        "shots": args.shots,
        "evaluation_id": args.evaluation_id,
        "evaluation_config": args.evaluation_config,
        "metric": "numeric_exact_match",
        "prompt_source": {k: v for k, v in args.meta_prompt.items() if k != "template"} if args.meta_prompt else None,
        "demonstration_train_indices": [] if args.meta_prompt else list(range(args.shots)),
        "condition": "sae_disabled" if country else "baseline",
        "country": country, "facet": facet, "n_samples": len(data), "correct": correct,
        "total": len(data), "accuracy": correct / len(data),
        "expansion": args.expansion,
        "disabled_features_by_layer": {f"L{k}R": v for k, v in (layer_features or {}).items()},
        "n_disabled_features": sum(map(len, (layer_features or {}).values())),
        "feature_files_by_layer": {f"L{k}R": str(feature_path(args, k)) for k in (layer_features or {})},
        "predictions_file": str(prediction_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[result] {country or 'baseline'}/{facet or ''}: {metrics['accuracy']:.4f} ({correct}/{len(data)})")
    return metrics


def save_summary(args, metrics):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json, summary_csv = summary_paths(args)
    summary_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    frame = pd.DataFrame(metrics)
    baseline_rows = frame.loc[frame["condition"] == "baseline"]
    if not baseline_rows.empty:
        baseline_accuracy = float(baseline_rows.iloc[0]["accuracy"])
        frame["baseline_accuracy"] = baseline_accuracy
        frame["accuracy_delta_vs_baseline"] = frame["accuracy"] - baseline_accuracy
        frame["accuracy_drop_vs_baseline"] = baseline_accuracy - frame["accuracy"]
    frame.to_csv(summary_csv, index=False)


def main():
    global base
    args = parse_args()
    groups = load_groups(args)
    for (country, facet), layers in groups.items():
        print(f"[plan] {country} / {facet}: {sum(map(len, layers.values()))} features across {len(layers)} layers")
    if args.list_conditions:
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("GSM8K loading requires the datasets package; install it in the project environment") from exc
    base = load_base()
    base.add_sae_paths(str(ROOT))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("openai/gsm8k", "main")
    data = dataset[args.split]
    if args.shots > len(dataset["train"]):
        raise ValueError(f"Requested {args.shots} shots but GSM8K train has only {len(dataset['train'])} examples")
    demonstrations = [] if args.meta_prompt else [dataset["train"][i] for i in range(args.shots)]
    if args.n_samples is not None:
        data = data.select(range(min(args.n_samples, len(data))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.dtype != "float32":
        raise ValueError("CPU execution requires --dtype float32")
    tokenizer, model = base.load_model_and_tokenizer(
        args.model_id, args.dtype, args.trust_remote_code, args.load_in_8bit, device,
    )
    summary_path, _ = summary_paths(args)
    metrics = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    def record(result):
        key = (result["condition"], result.get("country"), result.get("facet"))
        metrics[:] = [item for item in metrics if (item["condition"], item.get("country"), item.get("facet")) != key]
        metrics.append(result)
        save_summary(args, metrics)
    if not args.skip_baseline:
        record(evaluate(args, tokenizer, model, data, demonstrations))
    for (country, facet), layers in groups.items():
        record(evaluate(args, tokenizer, model, data, demonstrations, country, facet, layers))
    print(f"[done] {summary_paths(args)[1]}")


if __name__ == "__main__":
    main()
