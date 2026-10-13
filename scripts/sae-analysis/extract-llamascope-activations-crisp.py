import os
import glob
import argparse
import json
import shutil
import sys
import re
from pathlib import Path

import pandas as pd
import torch
from dotenv import dotenv_values
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPERIMENT_DIR = os.path.join(ROOT, "scripts", "experiment")
if EXPERIMENT_DIR not in sys.path:
    sys.path.append(EXPERIMENT_DIR)

ENV = dotenv_values(os.path.join(ROOT, ".env"))
DATA_DIR = Path(ENV["DATA_DIR"])

from generation_utils import chat_text
from utils import load_llamascope_sae


MODEL = ENV["GENERATION_MODEL_SHORT"]
JUDGE = ENV["JUDGE_MODEL_ID"]
GENERATION_MODEL_ID = ENV["GENERATION_MODEL_ID"]
MODEL_ID = ENV["SAE_BASE_MODEL_ID"]
SITE = ENV["SITE"]
EXPANSION = ENV["EXPANSION"]
TRUST_REMOTE_CODE = True


def parse_layers(raw: str) -> list[int]:
    layers = []
    for value in re.split(r"[,\s]+", str(raw).strip()):
        if not value:
            continue
        normalized = value.upper().removeprefix("L")
        for suffix in ("TC", "R", "A", "M"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        layers.append(int(normalized))
    if not layers:
        raise ValueError("LAYERS must contain at least one layer.")
    unique_layers = []
    seen = set()
    for layer in layers:
        if layer not in seen:
            seen.add(layer)
            unique_layers.append(layer)
    return unique_layers


LAYERS = parse_layers(ENV["LAYERS"])
INPUT_DIRS = [
    str(DATA_DIR / "analysis" / JUDGE / "with_name" / MODEL),
    str(DATA_DIR / "analysis" / JUDGE / "woname" / MODEL),
]
OUT = str(DATA_DIR / "sae" / JUDGE / MODEL)
GENERATIONS_GLOB = str(
    DATA_DIR
    / "generations"
    / "*"
    / f"{MODEL}_*.csv"
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--countries", nargs="+", help="Extract only these exact country names.")
    p.add_argument("--out_dir", type=Path, default=Path(OUT).parent, help="Output directory (defaults from DATA_DIR).")
    p.add_argument(
        "--tsv_glob",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional TSV glob(s). If set, these are added to files "
            "from the configured input directories."
        ),
    )

    p.add_argument(
        "--text_col",
        type=str,
        default="answer",
    )

    p.add_argument(
        "--use_prompt_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Tokenize the generation chat prompt plus --text_col. "
            "Use --no-use_prompt_answer to keep the old --text_col-only "
            "behavior."
        ),
    )

    p.add_argument(
        "--prompt_col",
        type=str,
        default="sys_prompt",
        help="Prompt column in generation CSVs.",
    )

    p.add_argument(
        "--question_col",
        type=str,
        default="question",
        help="Question column in generation CSVs.",
    )

    p.add_argument(
        "--prompt_answer_col",
        type=str,
        default="_prompt_answer_text",
        help="Internal column containing rendered chat prompt plus answer text.",
    )

    p.add_argument(
        "--prompt_answer_separator",
        type=str,
        default="",
        help=(
            "Optional separator inserted between rendered chat prompt "
            "and answer. Default empty matches generation continuation."
        ),
    )

    p.add_argument(
        "--chat_template_model_id",
        type=str,
        default=GENERATION_MODEL_ID,
        help=(
            "Tokenizer/model id whose chat template is used to reconstruct "
            "generation prompts."
        ),
    )

    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Total number of data shards. Use with --shard to run "
            "multiple independent extraction processes."
        ),
    )

    p.add_argument(
        "--shard",
        type=int,
        default=1,
        help=(
            "1-based shard number to process. For two shards, run "
            "--shard 1 and --shard 2."
        ),
    )

    p.add_argument(
        "--merge_shards_only",
        "--merge-shards-only",
        action="store_true",
        help=(
            "Merge completed shard parquet files into the unsharded "
            "layer output and exit without extracting activations."
        ),
    )

    p.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--flush_every",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--keep_parts",
        action="store_true",
        help=(
            "Keep intermediate OUT.parts parquet chunks after "
            "the final parquet is written."
        ),
    )

    p.add_argument(
        "--output_format",
        type=str,
        choices=["sparse", "dense"],
        default="sparse",
        help=(
            "sparse stores all positive SAE activations per token; "
            "dense stores the full [tokens x d_sae] activation "
            "matrix per row."
        ),
    )

    p.add_argument(
        "--no_store_tokens",
        action="store_true",
        help=(
            "Do not store token ids and decoded token strings "
            "in the output."
        ),
    )

    return p.parse_args()


def collect_tsv_paths(args):
    paths = []

    for input_dir in INPUT_DIRS:
        paths.extend(
            glob.glob(os.path.join(input_dir, "*.tsv"))
        )

    for pattern in args.tsv_glob or []:
        paths.extend(glob.glob(pattern))

    paths = sorted(set(paths))

    if not paths:
        searched = (
            list(INPUT_DIRS)
            + list(args.tsv_glob or [])
        )
        raise FileNotFoundError(
            f"No TSVs found. Searched: {searched}"
        )

    return paths


def collect_generation_paths():
    return sorted(
        set(glob.glob(GENERATIONS_GLOB))
    )


def infer_analysis_mode(path):
    return os.path.basename(
        os.path.dirname(
            os.path.dirname(path)
        )
    )


def validate_shard_args(args):
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")

    if args.shard < 1 or args.shard > args.num_shards:
        raise ValueError(
            "--shard must be between 1 and --num_shards "
            f"(got shard={args.shard}, num_shards={args.num_shards})."
        )


def shard_label(args):
    return shard_label_for(args.shard, args.num_shards)


def shard_label_for(shard, num_shards):
    if num_shards == 1:
        return None

    return f"{shard}of{num_shards}"


def split_parquet_path(path):
    ext = ".parquet"

    if path.endswith(ext):
        return path[:-len(ext)], ext

    return path, ext


def compact_generation_key(csv_path):
    """Map a generation CSV name to its facet/person analysis prefix."""

    path = Path(csv_path)
    stem = path.stem
    mode = path.parent.name
    prefix = f"{MODEL}_"
    suffix = f"_{mode}"

    if not stem.startswith(prefix):
        return None

    if not stem.endswith(suffix):
        return None

    return stem[
        len(prefix):-len(suffix)
    ]


def build_generation_indexes(paths):
    by_stem = {}
    by_mode_key = {}

    for path in paths:
        csv_path = Path(path)

        by_stem[csv_path.stem] = str(csv_path)

        key = compact_generation_key(csv_path)

        if key is None:
            continue

        mode = csv_path.parent.name
        by_mode_key.setdefault(
            (mode, key),
            [],
        ).append(str(csv_path))

    stems_longest_first = sorted(
        by_stem,
        key=len,
        reverse=True,
    )

    return by_stem, by_mode_key, stems_longest_first


def resolve_generation_path_for_source(
    source_file,
    analysis_mode,
    by_stem,
    by_mode_key,
    stems_longest_first,
):
    source_stem = Path(source_file).stem

    for generation_stem in stems_longest_first:
        if (
            source_stem == generation_stem
            or source_stem.startswith(f"{generation_stem}_")
        ):
            return by_stem[generation_stem]

    compact_key = source_stem.rsplit("_", 2)[0]
    candidates = by_mode_key.get(
        (analysis_mode, compact_key),
        [],
    )

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise ValueError(
            "Multiple generation CSVs match "
            f"source_file={source_file!r}, "
            f"analysis_mode={analysis_mode!r}: "
            f"{candidates}"
        )

    return None


def attach_generation_prompts(data, args, chat_template_tokenizer):
    if "answer_index" not in data.columns:
        raise ValueError(
            "Cannot recover sys_prompt because analysis TSVs do not "
            "contain answer_index."
        )

    generation_paths = collect_generation_paths()

    if not generation_paths:
        raise FileNotFoundError(
            "No generation CSVs found. Searched: "
            f"{GENERATIONS_GLOB}"
        )

    (
        by_stem,
        by_mode_key,
        stems_longest_first,
    ) = build_generation_indexes(generation_paths)

    source_map = {}
    missing_sources = []

    source_cols = [
        "source_file",
        "analysis_mode",
    ]

    for source_file, analysis_mode in (
        data[source_cols]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        generation_path = resolve_generation_path_for_source(
            source_file=source_file,
            analysis_mode=analysis_mode,
            by_stem=by_stem,
            by_mode_key=by_mode_key,
            stems_longest_first=stems_longest_first,
        )

        if generation_path is None:
            missing_sources.append(
                (source_file, analysis_mode)
            )
            continue

        source_map[(source_file, analysis_mode)] = generation_path

    if missing_sources:
        preview = missing_sources[:10]
        raise FileNotFoundError(
            "Could not match generation CSVs for "
            f"{len(missing_sources)} analysis source(s). "
            f"Examples: {preview}. "
            "Expected matching files under "
            f"{DATA_DIR / 'generations'}."
        )

    data = data.copy()
    data["generation_path"] = [
        source_map[(source_file, analysis_mode)]
        for source_file, analysis_mode in zip(
            data["source_file"],
            data["analysis_mode"],
        )
    ]
    data["generation_file"] = data["generation_path"].map(
        lambda path: Path(path).name
    )

    data[args.prompt_col] = pd.NA
    generation_question_col = "_generation_question"
    data[generation_question_col] = pd.NA

    for generation_path, group in data.groupby(
        "generation_path",
        sort=False,
    ):
        generation_df = pd.read_csv(generation_path)

        required_generation_cols = {
            args.prompt_col,
            args.question_col,
        }
        missing_generation_cols = (
            required_generation_cols - set(generation_df.columns)
        )

        if missing_generation_cols:
            raise ValueError(
                "Missing required column(s) "
                f"{sorted(missing_generation_cols)!r} in generation CSV: "
                f"{generation_path}"
            )

        answer_indices = pd.to_numeric(
            group["answer_index"],
            errors="coerce",
        )

        if answer_indices.isna().any():
            bad_rows = group.index[
                answer_indices.isna()
            ].tolist()[:10]
            raise ValueError(
                "Found non-numeric answer_index values in rows: "
                f"{bad_rows}"
            )

        answer_indices = answer_indices.astype(int)

        if (
            (answer_indices < 0).any()
            or (answer_indices >= len(generation_df)).any()
        ):
            raise IndexError(
                "answer_index out of range for generation CSV "
                f"{generation_path}; valid range is "
                f"0..{len(generation_df) - 1}."
            )

        generation_rows = generation_df.iloc[
            answer_indices.to_numpy()
        ]

        data.loc[group.index, args.prompt_col] = (
            generation_rows[args.prompt_col]
            .fillna("")
            .astype(str)
            .to_numpy()
        )
        data.loc[group.index, generation_question_col] = (
            generation_rows[args.question_col]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

    rendered_prompts = [
        chat_text(
            chat_template_tokenizer,
            system_text,
            question,
        )
        for system_text, question in zip(
            data[args.prompt_col].fillna("").astype(str),
            data[generation_question_col].fillna("").astype(str),
        )
    ]

    data[args.prompt_answer_col] = (
        pd.Series(rendered_prompts, index=data.index)
        + args.prompt_answer_separator
        + data[args.text_col].fillna("").astype(str)
    )

    print(
        "[info] Reconstructed chat-template prompts from "
        f"{data['generation_path'].nunique()} generation CSVs"
    )

    return data


def output_path_for_layer(
    out,
    layer,
    site,
    expansion,
    shard,
):
    if "{" in out:
        rendered = out.format(
            layer=layer,
            site=site,
            expansion=expansion,
            shard=shard or "",
        )
        root, ext = split_parquet_path(rendered)

        if shard and "{shard" not in out:
            return f"{root}_{shard}{ext}"

        return f"{root}{ext}"

    root, ext = split_parquet_path(out)

    parts = [
        root,
    ]

    if shard:
        parts.append(shard)

    parts.append(f"L{layer}{site}_{expansion}")

    return f"{'_'.join(parts)}{ext}"


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)

    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def ensure_new_outputs(paths):
    for path in paths:
        for candidate in (path, path + ".meta.json", path + ".parts"):
            if os.path.exists(candidate):
                raise FileExistsError(f"Output already exists: {candidate}. Choose another --out_dir.")


def write_part(buffer, parts_dir, part_idx):
    if not buffer:
        return part_idx

    out_path = os.path.join(
        parts_dir,
        f"part-{part_idx:06d}.parquet",
    )

    pd.DataFrame(buffer).to_parquet(
        out_path,
        index=False,
    )

    buffer.clear()

    return part_idx + 1


def write_metadata_json(
    out_path,
    args,
    layer,
    layers,
    d_model,
    d_sae,
):
    ensure_parent_dir(out_path)

    metadata = {
        "input_dirs": INPUT_DIRS,
        "countries": args.countries,
        "tsv_glob": args.tsv_glob,
        "model_id": MODEL_ID,
        "layer": layer,
        "layers": layers,
        "site": SITE,
        "expansion": EXPANSION,
        "num_shards": args.num_shards,
        "shard": args.shard,
        "shard_label": shard_label(args),
        "d_model": d_model,
        "d_sae": d_sae,
        "text_col": args.text_col,
        "activation_text_col": args.activation_text_col,
        "use_prompt_answer": args.use_prompt_answer,
        "tokenizer_add_special_tokens": not args.use_prompt_answer,
        "generations_glob": GENERATIONS_GLOB,
        "prompt_col": args.prompt_col,
        "question_col": args.question_col,
        "prompt_answer_format": (
            "chat_template_prompt_plus_answer"
            if args.use_prompt_answer
            else "answer_only"
        ),
        "chat_template_model_id": (
            args.chat_template_model_id
            if args.use_prompt_answer
            else None
        ),
        "prompt_answer_separator": args.prompt_answer_separator,
        "output_format": args.output_format,
        "stores_tokens": not args.no_store_tokens,
        "keep_parts": args.keep_parts,
        "activation_scope": (
            "all_positive_per_token"
            if args.output_format == "sparse"
            else "dense_per_token"
        ),
    }

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_json_atomic(path, payload):
    ensure_parent_dir(path)

    tmp_path = f"{path}.{os.getpid()}.tmp"

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(tmp_path, path)


def load_json_if_exists(path):
    if not os.path.exists(path):
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def merge_shard_outputs(args, layers):
    if args.num_shards < 2:
        raise ValueError(
            "--merge_shards_only requires --num_shards greater than 1."
        )

    ensure_new_outputs([
        output_path_for_layer(OUT, layer, SITE, EXPANSION, shard=None) for layer in layers
    ])
    for layer in layers:
        final_out_path = output_path_for_layer(
            OUT,
            layer,
            SITE,
            EXPANSION,
            shard=None,
        )

        shard_paths = [
            output_path_for_layer(
                OUT,
                layer,
                SITE,
                EXPANSION,
                shard=shard_label_for(
                    shard,
                    args.num_shards,
                ),
            )
            for shard in range(1, args.num_shards + 1)
        ]

        missing_paths = [
            path
            for path in shard_paths
            if not os.path.exists(path)
        ]

        if missing_paths:
            preview = missing_paths[:10]
            raise FileNotFoundError(
                "Cannot merge shards because shard output file(s) "
                f"are missing for layer {layer}: {preview}"
            )

        print(
            f"[info] Merging {len(shard_paths)} shard file(s) "
            f"for L{layer}{SITE}_{EXPANSION}"
        )

        merged = pd.concat(
            [
                pd.read_parquet(path)
                for path in shard_paths
            ],
            ignore_index=True,
        )

        sorted_by = None
        if "row_id" in merged.columns:
            merged = (
                merged
                .sort_values("row_id", kind="stable")
                .reset_index(drop=True)
            )
            sorted_by = "row_id"

        ensure_parent_dir(final_out_path)
        tmp_out_path = f"{final_out_path}.{os.getpid()}.tmp.parquet"
        merged.to_parquet(
            tmp_out_path,
            index=False,
        )
        os.replace(tmp_out_path, final_out_path)

        metadata_path = final_out_path + ".meta.json"
        first_shard_metadata_path = shard_paths[0] + ".meta.json"

        metadata = load_json_if_exists(
            first_shard_metadata_path
        )
        if not metadata:
            metadata = load_json_if_exists(metadata_path)
        metadata.update(
            {
                "num_shards": args.num_shards,
                "shard": None,
                "shard_label": None,
                "merged_from_shards": shard_paths,
                "n_merged_rows": int(len(merged)),
                "merge_sorted_by": sorted_by,
            }
        )
        write_json_atomic(
            metadata_path,
            metadata,
        )

        for shard_path in shard_paths:
            shard_metadata_path = shard_path + ".meta.json"
            if os.path.exists(shard_metadata_path):
                os.remove(shard_metadata_path)

        print(f"Saved merged L{layer}: {final_out_path}")
        print(f"Metadata: {metadata_path}")


def build_rows_for_layer(
    batch,
    feats,
    input_ids,
    enc_cpu,
    n_tokens,
    args,
):
    """Serialize one layer's activations with batched GPU-to-CPU transfers."""

    rows = []

    # One synchronization for the token lengths of the whole batch.
    n_tokens_cpu = (
        n_tokens
        .detach()
        .cpu()
        .tolist()
    )

    packed_chunks = []

    for batch_idx, nt in enumerate(n_tokens_cpu):
        if nt <= 0:
            continue

        # Only inspect actual tokens, excluding right padding.
        acts = feats[batch_idx, :nt, :]

        # Coordinates are [token_idx, feature_idx].
        coords = torch.nonzero(
            acts > 0,
            as_tuple=False,
        )

        if coords.numel() == 0:
            continue

        values = acts[
            coords[:, 0],
            coords[:, 1],
        ]

        batch_column = torch.full(
            size=(coords.shape[0], 1),
            fill_value=batch_idx,
            dtype=torch.float32,
            device=coords.device,
        )

        # Float32 represents the LlamaScope token/feature indices exactly.
        # packed columns:
        #   0: batch_idx
        #   1: token_idx
        #   2: feature_idx
        #   3: activation
        packed_chunk = torch.cat(
            [
                batch_column,
                coords.to(torch.float32),
                values.float().unsqueeze(1),
            ],
            dim=1,
        )

        packed_chunks.append(packed_chunk)

    sparse_rows = [
        {
            "feature_idx": [
                [] for _ in range(nt)
            ],
            "feature_act": [
                [] for _ in range(nt)
            ],
        }
        for nt in n_tokens_cpu
    ]

    if packed_chunks:
        # Single activation-related GPU -> CPU transfer
        # for the entire layer and batch.
        packed_cpu = (
            torch.cat(packed_chunks, dim=0)
            .detach()
            .cpu()
            .tolist()
        )

        for (
            batch_idx,
            token_idx,
            feature_idx,
            activation,
        ) in packed_cpu:
            batch_idx = int(batch_idx)
            token_idx = int(token_idx)
            feature_idx = int(feature_idx)

            sparse_rows[batch_idx]["feature_idx"][
                token_idx
            ].append(feature_idx)

            sparse_rows[batch_idx]["feature_act"][
                token_idx
            ].append(float(activation))


    for j, (_, r) in enumerate(batch.iterrows()):
        nt = int(n_tokens_cpu[j])

        row_out = {
            "check_country": r.get("check_country"),
            "country": r.get("country"),
            "name": r.get("name"),
            "answer_index": r.get("answer_index"),
            "generation_file": r.get("generation_file"),
            "generation_path": r.get("generation_path"),
            "label": int(r["label"]),
            "source_file": r.get("source_file"),
            "source_path": r.get("source_path"),
            "analysis_mode": r.get("analysis_mode"),
            "row_id": r.get("_row_id"),
            "n_tokens": nt,
        }

        if not args.no_store_tokens:
            row_out["token_ids"] = (
                input_ids[j, :nt]
                .tolist()
            )

            row_out["tokens"] = (
                enc_cpu.encodings[j]
                .tokens[:nt]
            )


        token_feature_idx = (
            sparse_rows[j]["feature_idx"]
        )

        token_feature_act = (
            sparse_rows[j]["feature_act"]
        )

        token_act_count = [
            len(values)
            for values in token_feature_act
        ]

        token_act_sum = [
            float(sum(values))
            for values in token_feature_act
        ]

        token_act_max = [
            float(max(values))
            if values
            else 0.0
            for values in token_feature_act
        ]

        row_out["token_feature_idx"] = (
            token_feature_idx
        )

        row_out["token_feature_act"] = (
            token_feature_act
        )

        row_out["token_act_count"] = (
            token_act_count
        )

        row_out["token_act_sum"] = (
            token_act_sum
        )

        row_out["token_act_max"] = (
            token_act_max
        )

        rows.append(row_out)

    return rows


def main():
    global OUT
    args = parse_args()
    validate_shard_args(args)
    OUT = str(args.out_dir / MODEL)

    args.activation_text_col = (
        args.prompt_answer_col
        if args.use_prompt_answer
        else args.text_col
    )

    layers = LAYERS

    if args.merge_shards_only:
        merge_shard_outputs(
            args,
            layers,
        )
        return

    current_shard_label = shard_label(args)
    ensure_new_outputs([
        output_path_for_layer(OUT, layer, SITE, EXPANSION, current_shard_label) for layer in layers
    ])

    out_parent = os.path.dirname(OUT)

    if out_parent:
        os.makedirs(
            out_parent,
            exist_ok=True,
        )


    paths = collect_tsv_paths(args)

    print(f"TSVs found={len(paths)}")

    dfs = []

    for path in paths:
        df = pd.read_csv(
            path,
            sep="\t",
        )

        df["source_file"] = os.path.basename(path)
        df["source_path"] = path
        df["analysis_mode"] = infer_analysis_mode(path)

        dfs.append(df)

    data = pd.concat(
        dfs,
        ignore_index=True,
    )
    if args.countries is not None:
        missing = set(args.countries) - set(data["country"].dropna())
        if missing:
            raise ValueError(f"No judge rows for exact country names: {sorted(missing)}")
        data = data[data["country"].isin(args.countries)].copy()

    # decision -> label
    data["decision"] = (
        data["decision"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["label"] = data["decision"].map(
        {
            "yes": 1,
            "no": 0,
        }
    )

    data = (
        data
        .dropna(
            subset=[
                "label",
                args.text_col,
            ]
        )
        .copy()
    )

    data["label"] = data["label"].astype(int)
    if data.empty:
        raise ValueError("No valid labelled responses to extract.")

    if args.use_prompt_answer:
        print(
            "[info] Loading chat-template tokenizer: "
            f"{args.chat_template_model_id}"
        )
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model_id,
            use_fast=True,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

        data = attach_generation_prompts(
            data,
            args,
            chat_template_tokenizer,
        )

        data = (
            data
            .dropna(
                subset=[
                    args.activation_text_col,
                ]
            )
            .copy()
        )

    data["_row_id"] = range(len(data))
    total_valid_rows = len(data)

    if args.num_shards > 1:
        data = (
            data
            .iloc[args.shard - 1::args.num_shards]
            .copy()
            .reset_index(drop=True)
        )

        if data.empty:
            raise ValueError(
                f"Shard {args.shard}of{args.num_shards} has no rows "
                f"from {total_valid_rows} valid row(s)."
            )

        print(
            "Valid rows="
            f"{total_valid_rows}; "
            f"processing shard {args.shard}of{args.num_shards} "
            f"rows={len(data)}"
        )
    else:
        print(f"Valid rows={len(data)}")


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.bfloat16
        if device.type == "cuda"
        else torch.float32
    )

    print(
        f"[info] device={device}, dtype={dtype}, "
        f"batch_size={args.batch_size}"
    )


    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    # build_rows_for_layer assumes valid tokens precede padding.
    tok.padding_side = "right"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    model.eval()


    saes = {}
    sae_dims = {}
    output_paths = {}
    parts_dirs = {}
    buffers = {}
    part_idxs = {}

    for layer in layers:
        sae, hp, top_k = load_llamascope_sae(
            layer=layer,
            site=SITE,
            expansion=EXPANSION,
            device=device,
            dtype=dtype,
        )

        d_model = int(hp["d_model"])
        d_sae = int(hp["d_sae"])

        saes[layer] = sae

        sae_dims[layer] = {
            "d_model": d_model,
            "d_sae": d_sae,
        }

        out_path = output_path_for_layer(
            OUT,
            layer,
            SITE,
            EXPANSION,
            current_shard_label,
        )

        output_paths[layer] = out_path
        parts_dirs[layer] = out_path + ".parts"
        buffers[layer] = []
        part_idxs[layer] = 0

        ensure_parent_dir(out_path)

        os.makedirs(
            parts_dirs[layer],
            exist_ok=False,
        )

        print(
            f"[info] LlamaScope SAE "
            f"L{layer}{SITE}-{EXPANSION}: "
            f"d_model={d_model} "
            f"d_sae={d_sae} "
            f"loaded_top_k={top_k}; "
            f"saving {args.output_format} token activations"
        )

        metadata_path = out_path + ".meta.json"

        write_metadata_json(
            metadata_path,
            args,
            layer=layer,
            layers=layers,
            d_model=d_model,
            d_sae=d_sae,
        )

        print(
            f"[info] Wrote metadata: "
            f"{metadata_path}"
        )


    captured = {}

    def make_hook(layer):
        def hook_fn(module, inp, out):
            hidden_states = (
                out[0]
                if isinstance(out, (tuple, list))
                else out
            )

            captured[layer] = hidden_states

        return hook_fn

    hook_handles = [
        model.model.layers[layer].register_forward_hook(
            make_hook(layer)
        )
        for layer in layers
    ]


    try:
        # inference_mode is slightly leaner than no_grad for pure inference.
        with torch.inference_mode():
            with tqdm(
                total=len(data),
                desc="docs",
                unit="doc",
            ) as pbar:
                for start in range(
                    0,
                    len(data),
                    args.batch_size,
                ):
                    batch = data.iloc[
                        start:start + args.batch_size
                    ]

                    texts = (
                        batch[args.activation_text_col]
                        .astype(str)
                        .tolist()
                    )

                    enc_cpu = tok(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                        add_special_tokens=not args.use_prompt_answer,
                    )

                    input_ids = enc_cpu.get("input_ids")
                    attention_mask_cpu = enc_cpu.get(
                        "attention_mask"
                    )

                    if attention_mask_cpu is None:
                        raise RuntimeError(
                            "Tokenizer output missing "
                            "attention_mask."
                        )

                    if input_ids is None:
                        raise RuntimeError(
                            "Tokenizer output missing "
                            "input_ids."
                        )

                    enc = {
                        key: value.to(
                            device,
                            non_blocking=True,
                        )
                        for key, value in enc_cpu.items()
                    }

                    attention_mask = enc["attention_mask"]

                    n_tokens = attention_mask.sum(
                        dim=1,
                    ).to(torch.int32)

                    captured.clear()

                    # Run the backbone directly to avoid allocating vocabulary logits.

                    model.model(
                        input_ids=enc["input_ids"],
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )

                    missing_layers = [
                        layer
                        for layer in layers
                        if layer not in captured
                    ]

                    if missing_layers:
                        raise RuntimeError(
                            "Hook did not capture hidden_states "
                            f"for layer(s): {missing_layers}"
                        )

                    for layer in layers:
                        hidden_states = captured[layer]

                        d_model = sae_dims[layer]["d_model"]

                        if hidden_states.shape[-1] != d_model:
                            raise RuntimeError(
                                f"Hook dim mismatch for "
                                f"layer {layer}: "
                                f"hs[-1]="
                                f"{hidden_states.shape[-1]} "
                                f"vs d_model={d_model}"
                            )

                        feats = saes[layer].encode(
                            hidden_states
                        )

                        layer_rows = build_rows_for_layer(
                            batch=batch,
                            feats=feats,
                            input_ids=input_ids,
                            enc_cpu=enc_cpu,
                            n_tokens=n_tokens,
                            args=args,
                        )

                        buffers[layer].extend(layer_rows)

                        if (
                            len(buffers[layer])
                            >= args.flush_every
                        ):
                            part_idxs[layer] = write_part(
                                buffers[layer],
                                parts_dirs[layer],
                                part_idxs[layer],
                            )

                        del feats

                    captured.clear()

                    pbar.update(len(batch))

    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()


    for layer in layers:
        if buffers[layer]:
            part_idxs[layer] = write_part(
                buffers[layer],
                parts_dirs[layer],
                part_idxs[layer],
            )


    for layer in layers:
        parts_dir = parts_dirs[layer]
        out_path = output_paths[layer]

        part_paths = sorted(
            glob.glob(
                os.path.join(
                    parts_dir,
                    "part-*.parquet",
                )
            )
        )

        if not part_paths:
            raise RuntimeError(
                f"No part files written for layer {layer}."
            )

        merged = pd.concat(
            [
                pd.read_parquet(path)
                for path in part_paths
            ],
            ignore_index=True,
        )

        merged.to_parquet(
            out_path,
            index=False,
        )

        print(f"Saved L{layer}: {out_path}")

        if args.keep_parts:
            print(
                f"Parts kept in: {parts_dir}"
            )
        else:
            shutil.rmtree(parts_dir)

            print(
                f"Deleted parts: {parts_dir}"
            )

    if args.num_shards > 1:
        try:
            merge_shard_outputs(args, layers)
        except FileNotFoundError:
            print(
                "[info] Not all shard outputs exist yet; "
                "skipping shard merge."
            )


if __name__ == "__main__":
    main()
