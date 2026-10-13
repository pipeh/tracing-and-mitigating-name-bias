#!/usr/bin/env python3

"""Extract DCLM token activations for the union of selected CRISP features.

Name and feature countries can be filtered independently. Dense columns
follow metadata feature_ids; sparse rows store feature IDs, positions, and
activations per token. One parquet and metadata sidecar are written per layer."""

import argparse
import ast
import glob
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import dotenv_values
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENV = dotenv_values(os.path.join(ROOT, ".env"))
DATA_DIR = Path(ENV["DATA_DIR"])

from utils import load_llamascope_sae


JUDGE_MODEL = ENV["JUDGE_MODEL_ID"]
MODEL_ID = ENV["SAE_BASE_MODEL_ID"]
SITE = ENV["SITE"]
EXPANSION = ENV["EXPANSION"]

DCLM_CSV_GLOB = str(DATA_DIR / "infinigram" / "v4_dclm-baseline_llama" / "*.csv")
NAMES_CSV = str(Path(ROOT) / "external" / "data" / "names.csv")
FEATURES_COL = "salient_features"
COUNTRY_COL = "country"
TEXT_COL = "text"
MAX_LENGTH = 1024
BATCH_SIZE = 16
FLUSH_EVERY = 500
OUTPUT_FORMAT = "sparse"
STORE_TOKENS = True

TARGET_COUNTRIES = [
    country.strip()
    for country in str(ENV.get("TARGET_COUNTRIES") or "").split(";")
    if country.strip()
]
TARGET_COUNTRY_KEYS = {country.casefold() for country in TARGET_COUNTRIES}


def parse_layers(raw: str) -> List[int]:
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


def parse_country_list(value: str) -> List[str]:
    countries = [country.strip() for country in value.split(";") if country.strip()]
    if not countries:
        raise argparse.ArgumentTypeError("Provide at least one semicolon-separated country.")
    return list(dict.fromkeys(countries))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract token-level DCLM activations for CRISP-selected "
            "LlamaScope SAE features."
        )
    )
    p.add_argument("--csv_glob", type=str, default=DCLM_CSV_GLOB)
    p.add_argument("--names_csv", type=str, default=NAMES_CSV)
    p.add_argument("--salient_glob", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument(
        "--name_countries", type=parse_country_list, default=None,
        help="Countries whose names/documents to process, separated by semicolons. Overrides TARGET_COUNTRIES for names only.",
    )
    p.add_argument(
        "--feature_countries", type=parse_country_list, default=None,
        help="Countries whose CRISP feature rows to use, separated by semicolons. Overrides TARGET_COUNTRIES for features only.",
    )
    p.add_argument("--text_col", type=str, default=TEXT_COL)
    p.add_argument("--features_col", type=str, default=FEATURES_COL)
    p.add_argument("--country_col", type=str, default=COUNTRY_COL)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--flush_every", type=int, default=FLUSH_EVERY)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use, e.g. cuda:0 or cuda:1. Defaults to cuda if available.",
    )
    p.add_argument(
        "--output_format",
        choices=["dense", "sparse"],
        default=OUTPUT_FORMAT,
        help=(
            "dense writes a raw [tokens x selected_features] activation matrix "
            "per document; sparse writes only active selected features per token."
        ),
    )
    p.add_argument(
        "--store_tokens",
        dest="store_tokens",
        action="store_true",
        default=STORE_TOKENS,
    )
    p.add_argument(
        "--no_store_tokens",
        dest="store_tokens",
        action="store_false",
    )
    p.add_argument(
        "--all_countries",
        action="store_true",
        help="Ignore TARGET_COUNTRIES and process every country in --names_csv.",
    )
    p.add_argument(
        "--keep_parts",
        action="store_true",
        help="Keep intermediate parquet parts after the final layer parquet is written.",
    )
    p.add_argument(
        "--merge_shards_only",
        "--merge-shards-only",
        action="store_true",
        help=(
            "Merge completed shard parquet files into the unsharded layer outputs "
            "and exit without loading the model."
        ),
    )

    args = p.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.flush_every < 1:
        raise ValueError("--flush_every must be >= 1")
    return args


def default_salient_glob(layer: int) -> str:
    return str(
        DATA_DIR
        / "crisp"
        / JUDGE_MODEL
        / f"topic_country_salient_features_*_L{layer}{SITE}_{EXPANSION}.parquet"
    )


def salient_glob_for_layer(args, layer: int) -> str:
    if args.salient_glob is None:
        return default_salient_glob(layer)
    if "{" in args.salient_glob:
        return args.salient_glob.format(
            layer=layer,
            site=SITE,
            expansion=EXPANSION,
            judge_model=JUDGE_MODEL,
        )
    return args.salient_glob


def output_dir(args) -> str:
    if args.out_dir:
        return args.out_dir
    return str(DATA_DIR / "sae-salient-tokens" / JUDGE_MODEL)


def output_path_for_layer(args, layer: int) -> str:
    return os.path.join(
        output_dir(args),
        f"llamascope_salient_token_acts_dclm_L{layer}{SITE}_{EXPANSION}.parquet",
    )


def shard_output_path_for_layer(args, layer: int, shard_id: int) -> str:
    out_path = output_path_for_layer(args, layer)
    stem, ext = os.path.splitext(out_path)
    return f"{stem}.{shard_label_for(shard_id, args.num_shards)}{ext}"


def shard_label_for(shard_id: int, num_shards: int) -> str:
    return f"shard{shard_id:02d}-of-{num_shards - 1:02d}"


def shard_label(args) -> str:
    return shard_label_for(args.shard_id, args.num_shards)


def active_country_keys(args) -> Set[str]:
    """Filter CRISP rows independently of the names being processed."""
    if args.feature_countries is not None:
        return {country.casefold() for country in args.feature_countries}
    if args.all_countries:
        return set()
    return set(TARGET_COUNTRY_KEYS)


def active_name_country_keys(args) -> Set[str]:
    if args.name_countries is not None:
        return {country.casefold() for country in args.name_countries}
    if args.all_countries:
        return set()
    return set(TARGET_COUNTRY_KEYS)


def selected_countries_for_metadata(args) -> List[str]:
    if args.name_countries is not None:
        return list(args.name_countries)
    if args.all_countries:
        return []
    return list(TARGET_COUNTRIES)


def selected_feature_countries_for_metadata(args) -> List[str]:
    if args.feature_countries is not None:
        return list(args.feature_countries)
    if args.all_countries:
        return []
    return list(TARGET_COUNTRIES)


@dataclass
class LayerContext:
    layer: int
    salient_glob: str
    salient_paths: List[str]
    salient_stats: Dict
    feature_ids: List[int]
    feature_id_tensor: torch.Tensor
    encoder_weight: torch.Tensor
    encoder_bias: Optional[torch.Tensor]
    sae: object
    d_model: int
    d_sae: int
    out_path: str
    parts_dir: str
    metadata_path: str
    buffer: List[Dict] = field(default_factory=list)
    part_idx: int = 0


def parse_feature_sequence(value) -> Iterable[int]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        value = ast.literal_eval(text)
    if isinstance(value, (list, tuple, set)):
        return [int(feature) for feature in value]
    if hasattr(value, "tolist"):
        return [int(feature) for feature in value.tolist()]
    raise TypeError(f"Unsupported salient feature value: {type(value)!r}")


def load_salient_feature_union(
    paths: Sequence[str],
    features_col: str,
    country_col: str,
    country_keys: Set[str],
) -> Tuple[List[int], Dict]:
    features: Set[int] = set()
    rows_total = 0
    rows_selected = 0
    countries_seen: Set[str] = set()
    countries_selected: Set[str] = set()

    for path in paths:
        df = pd.read_parquet(path)
        if features_col not in df.columns:
            raise ValueError(f"{path} is missing required column: {features_col}")

        rows_total += len(df)
        if country_col in df.columns:
            country_values = df[country_col].fillna("").astype(str).str.strip()
            countries_seen.update(country for country in country_values if country)
            if country_keys:
                mask = country_values.str.casefold().isin(country_keys)
                df = df.loc[mask]
                countries_selected.update(country for country in country_values[mask] if country)
            else:
                countries_selected.update(country for country in country_values if country)
        elif country_keys:
            raise ValueError(
                f"{path} has no {country_col!r} column, so TARGET_COUNTRIES cannot "
                "be applied to salient features."
            )

        rows_selected += len(df)
        for value in df[features_col]:
            features.update(parse_feature_sequence(value))

    feature_ids = sorted(features)
    if not feature_ids:
        raise ValueError(
            "No salient features found after applying the country filter. "
            f"Matched files: {list(paths)}"
        )

    stats = {
        "n_salient_rows_total": rows_total,
        "n_salient_rows_selected": rows_selected,
        "countries_seen_in_salient": sorted(countries_seen),
        "countries_selected_in_salient": sorted(countries_selected),
    }
    return feature_ids, stats


def load_name_metadata(
    names_csv: str,
    country_keys: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict]:
    df = pd.read_csv(names_csv)
    required = {"name", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{names_csv} is missing required column(s): {sorted(missing)}")

    df = df.copy()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    df = df[(df["name"] != "") & (df["country"] != "")]

    if country_keys:
        df = df[df["country"].str.casefold().isin(country_keys)]

    if df.empty:
        raise ValueError(
            "No names matched the requested country filter. "
            f"names_csv={names_csv!r}, TARGET_COUNTRIES={TARGET_COUNTRIES!r}"
        )

    name_countries = df.groupby("name")["country"].unique()
    ambiguous = {
        str(name): sorted(str(country) for country in countries)
        for name, countries in name_countries.items()
        if len(countries) > 1
    }
    if ambiguous:
        preview = dict(list(ambiguous.items())[:10])
        raise ValueError(
            "Some names map to multiple selected countries, so DCLM CSVs cannot "
            f"be labeled unambiguously. Examples: {preview}"
        )

    name_to_country = {
        str(name): str(countries[0])
        for name, countries in name_countries.items()
    }

    names_by_country: Dict[str, List[str]] = {}
    for country, group in df.groupby("country", sort=True):
        names_by_country[str(country)] = sorted(group["name"].unique().tolist())

    stats = {
        "names_csv": names_csv,
        "n_name_rows_selected": int(len(df)),
        "n_unique_names_selected": len(name_to_country),
        "names_by_country": names_by_country,
    }
    return name_to_country, names_by_country, stats


def collect_filtered_csv_paths(
    csv_glob: str,
    name_to_country: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    all_paths = sorted(glob.glob(csv_glob))
    if not all_paths:
        raise FileNotFoundError(f"No CSVs matched: {csv_glob}")

    path_by_name = {Path(path).stem: path for path in all_paths}
    selected = [
        path_by_name[name]
        for name in sorted(name_to_country)
        if name in path_by_name
    ]
    missing_names = [
        name
        for name in sorted(name_to_country)
        if name not in path_by_name
    ]

    if not selected:
        raise FileNotFoundError(
            "No DCLM CSV files matched the selected names. "
            f"csv_glob={csv_glob!r}"
        )

    return selected, missing_names


def write_json_atomic(path: str, payload: Dict):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def write_part(buffer: List[Dict], parts_dir: str, part_idx: int) -> int:
    if not buffer:
        return part_idx
    os.makedirs(parts_dir, exist_ok=True)
    part_path = os.path.join(parts_dir, f"part-{part_idx:06d}.parquet")
    pd.DataFrame(buffer).to_parquet(part_path, index=False)
    buffer.clear()
    return part_idx + 1


def merge_parts(parts_dir: str, out_path: str):
    part_paths = sorted(glob.glob(os.path.join(parts_dir, "part-*.parquet")))
    if not part_paths:
        raise RuntimeError(f"No part files written under: {parts_dir}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged = pd.concat(
        [pd.read_parquet(part_path) for part_path in part_paths],
        ignore_index=True,
    )
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    merged.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def merge_shards_for_layer(args, layer: int):
    if args.num_shards < 2:
        raise ValueError("--merge_shards_only requires --num_shards >= 2")

    out_path = output_path_for_layer(args, layer)
    shard_paths = [
        shard_output_path_for_layer(args, layer, shard_id)
        for shard_id in range(args.num_shards)
    ]
    missing = [path for path in shard_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Cannot merge because some shard parquet files are missing: "
            f"{missing}"
        )

    metadata_paths = [path + ".meta.json" for path in shard_paths]
    missing_metadata = [
        path for path in metadata_paths if not os.path.exists(path)
    ]
    if missing_metadata:
        raise FileNotFoundError(
            "Cannot merge because some shard metadata files are missing: "
            f"{missing_metadata}"
        )

    print(f"[info] Merging L{layer}{SITE}-{EXPANSION} shards -> {out_path}")
    merged = pd.concat(
        [pd.read_parquet(path) for path in shard_paths],
        ignore_index=True,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.{os.getpid()}.tmp"
    merged.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)

    shard_metadata = []
    for path in metadata_paths:
        with open(path, "r", encoding="utf-8") as f:
            shard_metadata.append(json.load(f))

    consistency_keys = [
        "model_id",
        "judge_model",
        "layer",
        "site",
        "expansion",
        "output_format",
        "store_tokens",
        "activation_scope",
        "feature_ids",
        "target_countries",
        "feature_countries",
        "all_countries",
    ]
    reference_metadata = shard_metadata[0]
    for shard_id, metadata in enumerate(shard_metadata[1:], start=1):
        mismatched = [
            key
            for key in consistency_keys
            if metadata.get(key) != reference_metadata.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Shard metadata mismatch for shard {shard_id}: {mismatched}"
            )

    merged_metadata = dict(reference_metadata)
    csv_paths = []
    for metadata in shard_metadata:
        csv_paths.extend(metadata.get("csv_paths", []))

    merged_metadata.update(
        {
            "out": out_path,
            "parts_dir": None,
            "num_shards": args.num_shards,
            "shard_id": None,
            "merged_from_shards": shard_paths,
            "merged_from_metadata": metadata_paths,
            "csv_paths": csv_paths,
            "n_csv_paths": len(csv_paths),
            "n_rows": int(len(merged)),
        }
    )
    write_json_atomic(out_path + ".meta.json", merged_metadata)
    print(f"[info] Saved merged parquet: {out_path}")
    print(f"[info] Saved merged metadata: {out_path}.meta.json")


def transformer_layer(model, layer: int):
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise TypeError("Expected a Llama-style model exposing model.model.layers.") from exc

    if layer < 0 or layer >= len(layers):
        raise ValueError(f"Layer {layer} is outside model range [0, {len(layers) - 1}].")
    return layers[layer]


def apply_selected_sae_activation(
    sae,
    pre_activations: torch.Tensor,
    feature_ids: torch.Tensor,
) -> torch.Tensor:
    activation = sae.act

    if hasattr(activation, "get_jumprelu_threshold"):
        thresholds = activation.get_jumprelu_threshold().reshape(-1)
        thresholds = thresholds.index_select(0, feature_ids).to(
            device=pre_activations.device,
            dtype=pre_activations.dtype,
        )
        return pre_activations * (pre_activations > thresholds)

    if hasattr(activation, "log_jumprelu_threshold"):
        thresholds = activation.log_jumprelu_threshold.exp().reshape(-1)
        thresholds = thresholds.index_select(0, feature_ids).to(
            device=pre_activations.device,
            dtype=pre_activations.dtype,
        )
        return pre_activations * (pre_activations > thresholds)

    activated = activation(pre_activations)
    if activated.shape != pre_activations.shape:
        raise ValueError(
            "SAE activation changed selected feature activation shape: "
            f"{tuple(pre_activations.shape)} -> {tuple(activated.shape)}"
        )
    return activated


def selected_sae_encode(
    sae,
    hidden_states: torch.Tensor,
    feature_ids: torch.Tensor,
    encoder_weight: torch.Tensor,
    encoder_bias: Optional[torch.Tensor],
):
    pre_activations = F.linear(hidden_states, encoder_weight, encoder_bias)
    return apply_selected_sae_activation(sae, pre_activations, feature_ids)


def dense_token_acts_batch(
    acts: torch.Tensor,
    n_tokens_cpu: Sequence[int],
) -> List[Dict[str, List]]:
    vals = acts.detach().cpu().float()
    return [
        {"acts_TxF": vals[j, : int(nt), :].tolist()}
        for j, nt in enumerate(n_tokens_cpu)
    ]


def sparse_token_acts_batch(
    acts: torch.Tensor,
    n_tokens: torch.Tensor,
    n_tokens_cpu: Sequence[int],
    feature_ids: Sequence[int],
) -> List[Dict[str, List]]:
    batch_size, max_tokens, _n_features = acts.shape
    token_mask = (
        torch.arange(max_tokens, device=acts.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        < n_tokens.unsqueeze(1)
    )
    active = (acts > 0) & token_mask.unsqueeze(-1)
    coords = active.nonzero(as_tuple=False)

    rows = [
        {
            "token_feature_idx": [[] for _ in range(int(nt))],
            "token_feature_pos": [[] for _ in range(int(nt))],
            "token_feature_act": [[] for _ in range(int(nt))],
        }
        for nt in n_tokens_cpu
    ]

    if coords.numel() == 0:
        return rows

    values = acts[coords[:, 0], coords[:, 1], coords[:, 2]]
    packed = torch.cat(
        [
            coords.to(torch.float32),
            values.float().unsqueeze(1),
        ],
        dim=1,
    )

    for batch_idx, token_idx, feature_pos, activation in packed.detach().cpu().tolist():
        batch_idx = int(batch_idx)
        token_idx = int(token_idx)
        feature_pos = int(feature_pos)
        rows[batch_idx]["token_feature_pos"][token_idx].append(feature_pos)
        rows[batch_idx]["token_feature_idx"][token_idx].append(
            int(feature_ids[feature_pos])
        )
        rows[batch_idx]["token_feature_act"][token_idx].append(float(activation))

    return rows


def serialize_token_acts_batch(
    acts: torch.Tensor,
    n_tokens: torch.Tensor,
    n_tokens_cpu: Sequence[int],
    feature_ids: Sequence[int],
    output_format: str,
) -> List[Dict[str, List]]:
    if output_format == "dense":
        return dense_token_acts_batch(acts, n_tokens_cpu)
    return sparse_token_acts_batch(acts, n_tokens, n_tokens_cpu, feature_ids)


def metadata_payload(
    args,
    layer: int,
    out_path: str,
    parts_dir: str,
    salient_paths: Sequence[str],
    salient_glob: str,
    salient_stats: Dict,
    feature_ids: Sequence[int],
    csv_paths: Sequence[str],
    missing_names: Sequence[str],
    names_stats: Dict,
    d_model: int,
    d_sae: int,
    device,
) -> Dict:
    payload = {
        "model_id": MODEL_ID,
        "judge_model": JUDGE_MODEL,
        "layer": layer,
        "layers": LAYERS,
        "site": SITE,
        "expansion": EXPANSION,
        "d_model": d_model,
        "d_sae": d_sae,
        "device": str(device),
        "output_format": args.output_format,
        "store_tokens": args.store_tokens,
        "activation_scope": "target_country_crisp_salient_features_per_token",
        "feature_ids": list(feature_ids),
        "n_features": len(feature_ids),
        "feature_alignment": (
            "Dense acts_TxF[token_position][feature_position] is aligned "
            "with metadata feature_ids. Sparse token_feature_idx contains "
            "actual SAE feature IDs and token_feature_pos contains positions "
            "in feature_ids."
        ),
        "target_countries": selected_countries_for_metadata(args),
        "feature_countries": selected_feature_countries_for_metadata(args),
        "all_countries": bool(args.all_countries),
        "country_filter_col": args.country_col,
        "features_col": args.features_col,
        "text_col": args.text_col,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "flush_every": args.flush_every,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "csv_glob": args.csv_glob,
        "csv_paths": list(csv_paths),
        "n_csv_paths": len(csv_paths),
        "missing_selected_name_csvs": list(missing_names),
        "n_missing_selected_name_csvs": len(missing_names),
        "salient_glob": salient_glob,
        "salient_paths": list(salient_paths),
        "n_salient_paths": len(salient_paths),
        "out": out_path,
        "parts_dir": parts_dir,
    }
    payload.update(salient_stats)
    payload.update(names_stats)
    return payload


def drop_unused_sae_modules(sae):
    if hasattr(sae, "enc"):
        sae.enc = torch.nn.Identity()
    if hasattr(sae, "dec"):
        sae.dec = torch.nn.Identity()


def prepare_layer_context(
    args,
    csv_paths: Sequence[str],
    missing_names: Sequence[str],
    names_stats: Dict,
    device,
    dtype,
    layer: int,
) -> LayerContext:
    salient_glob = salient_glob_for_layer(args, layer)
    salient_paths = sorted(glob.glob(salient_glob))
    if not salient_paths:
        raise FileNotFoundError(f"No salient feature files matched: {salient_glob}")

    feature_ids, salient_stats = load_salient_feature_union(
        salient_paths,
        features_col=args.features_col,
        country_col=args.country_col,
        country_keys=active_country_keys(args),
    )

    out_path = output_path_for_layer(args, layer)
    if args.num_shards == 1:
        layer_out_path = out_path
        parts_dir = out_path + ".parts"
    else:
        stem, _ext = os.path.splitext(out_path)
        layer_out_path = shard_output_path_for_layer(args, layer, args.shard_id)
        parts_dir = f"{stem}.{shard_label(args)}.parts"

    if os.path.exists(parts_dir):
        shutil.rmtree(parts_dir)
    os.makedirs(parts_dir, exist_ok=True)

    sae, hp, _top_k = load_llamascope_sae(
        layer=layer,
        site=SITE,
        expansion=EXPANSION,
        device=device,
        dtype=dtype,
    )
    d_model = int(hp["d_model"])
    d_sae = int(hp["d_sae"])

    invalid_features = [
        feature_id for feature_id in feature_ids if feature_id < 0 or feature_id >= d_sae
    ]
    if invalid_features:
        raise ValueError(
            f"{len(invalid_features)} feature id(s) are outside SAE dim {d_sae}: "
            f"{invalid_features[:20]}"
        )

    metadata_path = layer_out_path + ".meta.json"
    feature_id_tensor = torch.tensor(feature_ids, device=device, dtype=torch.long)
    encoder_weight = sae.enc.weight.index_select(0, feature_id_tensor).contiguous()
    encoder_bias = None
    if sae.enc.bias is not None:
        encoder_bias = sae.enc.bias.index_select(0, feature_id_tensor).contiguous()
    drop_unused_sae_modules(sae)

    context = LayerContext(
        layer=layer,
        salient_glob=salient_glob,
        salient_paths=salient_paths,
        salient_stats=salient_stats,
        feature_ids=feature_ids,
        feature_id_tensor=feature_id_tensor,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        sae=sae,
        d_model=d_model,
        d_sae=d_sae,
        out_path=layer_out_path,
        parts_dir=parts_dir,
        metadata_path=metadata_path,
    )

    write_json_atomic(
        metadata_path,
        metadata_payload(
            args=args,
            layer=layer,
            out_path=layer_out_path,
            parts_dir=parts_dir,
            salient_paths=salient_paths,
            salient_glob=salient_glob,
            salient_stats=salient_stats,
            feature_ids=feature_ids,
            csv_paths=csv_paths,
            missing_names=missing_names,
            names_stats=names_stats,
            d_model=d_model,
            d_sae=d_sae,
            device=device,
        ),
    )

    print(
        f"[info] L{layer}{SITE}-{EXPANSION}: {len(feature_ids)} salient feature(s), "
        f"{len(csv_paths)} DCLM CSV(s), output={layer_out_path}"
    )
    return context


def make_activation_hook(captured: Dict[int, torch.Tensor], context: LayerContext):
    def hook_fn(_module, _inp, out):
        hidden_states = out[0] if isinstance(out, (tuple, list)) else out
        if hidden_states.shape[-1] != context.d_model:
            raise RuntimeError(
                f"L{context.layer} hook dim mismatch: "
                f"hs[-1]={hidden_states.shape[-1]} vs d_model={context.d_model}"
            )
        captured[context.layer] = selected_sae_encode(
            context.sae,
            hidden_states,
            context.feature_id_tensor,
            context.encoder_weight,
            context.encoder_bias,
        )

    return hook_fn


def build_base_rows(batch, enc_cpu, n_tokens, source_file: str, name: str, country: str, args):
    n_tokens_cpu = n_tokens.detach().cpu().tolist()
    rows = []
    for j, (_, row) in enumerate(batch.iterrows()):
        nt = int(n_tokens_cpu[j])
        row_out = {
            "country": country,
            "name": name,
            "doc_ix": row.get("doc_ix"),
            "doc_len": row.get("doc_len"),
            "disp_len": row.get("disp_len"),
            "source_file": source_file,
            "n_tokens": nt,
        }
        if "lang" in row.index:
            row_out["lang"] = row.get("lang")
        if args.store_tokens:
            row_out["token_ids"] = enc_cpu["input_ids"][j, :nt].tolist()
            row_out["tokens"] = enc_cpu.encodings[j].tokens[:nt]
        rows.append(row_out)
    return rows, n_tokens_cpu


def flush_layer_buffers(contexts: Sequence[LayerContext]):
    for context in contexts:
        if context.buffer:
            context.part_idx = write_part(
                context.buffer,
                context.parts_dir,
                context.part_idx,
            )


def finalize_layer_outputs(args, contexts: Sequence[LayerContext]):
    for context in contexts:
        merge_parts(context.parts_dir, context.out_path)
        if not args.keep_parts:
            shutil.rmtree(context.parts_dir)

        print(f"[info] Saved: {context.out_path}")
        print(f"[info] Metadata: {context.metadata_path}")


def process_all_layers(
    args,
    csv_paths: Sequence[str],
    missing_names: Sequence[str],
    name_to_country: Dict[str, str],
    names_stats: Dict,
    tokenizer,
    model,
    device,
    dtype,
):
    contexts = [
        prepare_layer_context(
            args=args,
            csv_paths=csv_paths,
            missing_names=missing_names,
            names_stats=names_stats,
            device=device,
            dtype=dtype,
            layer=layer,
        )
        for layer in LAYERS
    ]

    captured: Dict[int, torch.Tensor] = {}
    hook_handles = []
    try:
        for context in contexts:
            hook_handles.append(
                transformer_layer(model, context.layer).register_forward_hook(
                    make_activation_hook(captured, context)
                )
            )

        with torch.inference_mode():
            for csv_path in tqdm(csv_paths, desc="csv files", unit="file"):
                source_file = os.path.basename(csv_path)
                name = Path(source_file).stem
                country = name_to_country[name]

                df = pd.read_csv(csv_path)
                if args.text_col not in df.columns:
                    raise ValueError(f"{csv_path} is missing text column {args.text_col!r}")

                for start in tqdm(
                    range(0, len(df), args.batch_size),
                    desc=source_file,
                    leave=False,
                    unit="batch",
                ):
                    batch = df.iloc[start:start + args.batch_size]
                    texts = batch[args.text_col].fillna("").astype(str).tolist()

                    enc_cpu = tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_length,
                    )
                    enc = {key: value.to(device) for key, value in enc_cpu.items()}

                    attn = enc.get("attention_mask")
                    if attn is None:
                        raise RuntimeError("Tokenizer output missing attention_mask.")
                    n_tokens = attn.sum(dim=1).to(torch.int32)

                    captured.clear()
                    _ = model(**enc)
                    base_rows, n_tokens_cpu = build_base_rows(
                        batch=batch,
                        enc_cpu=enc_cpu,
                        n_tokens=n_tokens,
                        source_file=source_file,
                        name=name,
                        country=country,
                        args=args,
                    )

                    for context in contexts:
                        acts = captured.get(context.layer)
                        if acts is None:
                            raise RuntimeError(
                                f"Hook did not capture activations for layer {context.layer}."
                            )

                        stats_rows = serialize_token_acts_batch(
                            acts=acts,
                            n_tokens=n_tokens,
                            n_tokens_cpu=n_tokens_cpu,
                            feature_ids=context.feature_ids,
                            output_format=args.output_format,
                        )

                        for base_row, stats in zip(base_rows, stats_rows):
                            row_out = dict(base_row)
                            row_out.update(stats)
                            context.buffer.append(row_out)

                        if len(context.buffer) >= args.flush_every:
                            context.part_idx = write_part(
                                context.buffer,
                                context.parts_dir,
                                context.part_idx,
                            )
                        captured[context.layer] = None
                        del acts

                    captured.clear()
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()

    flush_layer_buffers(contexts)
    finalize_layer_outputs(args, contexts)


def main():
    args = parse_args()

    if args.merge_shards_only:
        for layer in LAYERS:
            merge_shards_for_layer(args, layer)
        return

    name_to_country, _names_by_country, names_stats = load_name_metadata(
        args.names_csv,
        country_keys=active_name_country_keys(args),
    )
    csv_paths, missing_names = collect_filtered_csv_paths(args.csv_glob, name_to_country)
    csv_paths = [
        path
        for idx, path in enumerate(csv_paths)
        if idx % args.num_shards == args.shard_id
    ]
    if not csv_paths:
        raise RuntimeError(
            f"Shard {args.shard_id}/{args.num_shards} has no selected DCLM CSVs."
        )

    if missing_names:
        print(
            f"[warn] {len(missing_names)} selected name(s) have no matching DCLM CSV; "
            f"examples: {missing_names[:10]}"
        )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    process_all_layers(
        args=args,
        csv_paths=csv_paths,
        missing_names=missing_names,
        name_to_country=name_to_country,
        names_stats=names_stats,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
