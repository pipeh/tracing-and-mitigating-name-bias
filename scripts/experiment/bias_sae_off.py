"""Compute SAE-off default-bias summaries by country or by name."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
ENV = dotenv_values(ROOT / ".env")
DATA_DIR = Path(ENV["DATA_DIR"])

ANALYSIS_DIR = DATA_DIR / "analysis-sae-off"
JUDGE_MODEL = ENV["JUDGE_MODEL_ID"]
MODEL = ENV["GENERATION_MODEL_SHORT"]
PREFERRED_TOPIC_ORDER = ("food", "clothing", "tradition", "rituals")
SUPPORTED_MODES = ("name_and_country", "with_country", "with_name", "woname")


@dataclass(frozen=True)
class FileSpec:
    topic: str
    mode: str
    layer: str
    expansion: str


def decision_to_binary(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
        }
    )


def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "decision" not in df.columns:
        raise ValueError(f"{path} missing required column: decision")

    df = df.copy()
    df["decision_binary"] = decision_to_binary(df["decision"])
    invalid = int(df["decision_binary"].isna().sum())
    if invalid:
        print(f"[warn] {path}: dropping {invalid} invalid decisions")
        df = df.dropna(subset=["decision_binary"])

    df["decision_binary"] = df["decision_binary"].astype(int)
    return df


def model_dir(root: Path, mode: str, model: str) -> Path:
    path = root / mode / model
    if not path.exists():
        raise FileNotFoundError(f"Missing directory for {mode}/{model}: {path}")
    return path


def parse_sae_off_file(path: Path, model: str) -> FileSpec:
    prefix = f"{model}_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"file does not start with {prefix!r}")

    remainder = path.stem[len(prefix):]
    if "_FirstPerson_" not in remainder:
        raise ValueError("missing '_FirstPerson_'")

    topic, tail = remainder.split("_FirstPerson_", 1)
    mode_pattern = "|".join(re.escape(mode) for mode in SUPPORTED_MODES)
    layer_pattern = r"L\d+(?:TC|R|A|M)"
    config_pattern = (
        rf"(?:{layer_pattern}|"
        rf"multilayer_per_layer_{layer_pattern}(?:-{layer_pattern})*)"
    )
    match = re.fullmatch(
        rf".+_(?P<mode>{mode_pattern})_"
        rf"(?P<layer>{config_pattern})_"
        rf"(?P<expansion>\d+x)"
        rf"(?:_top(?P<num_features>\d+))?_sae_off"
        rf"(?:_random_ablation_(?P<random_seed>\d+))?(?:_.+)?",
        tail,
    )
    if not match:
        raise ValueError("could not parse mode/layer/expansion")

    layer = match.group("layer")
    if match.group("num_features"):
        layer = f"{layer}_top{match.group('num_features')}"
    if match.group("random_seed") is not None:
        layer = f"{layer}_random_ablation_{match.group('random_seed')}"

    return FileSpec(
        topic=topic.strip().lower(),
        mode=match.group("mode"),
        layer=layer,
        expansion=match.group("expansion"),
    )


def load_mode(
    mode_dir: Path,
    model: str,
    expected_mode: str,
    group_by: str,
) -> pd.DataFrame:
    rows = []

    for path in sorted(mode_dir.glob("*.tsv")):
        try:
            spec = parse_sae_off_file(path, model)
        except ValueError as exc:
            print(f"[warn] skipping {path.name}: {exc}")
            continue

        if spec.mode != expected_mode:
            print(f"[warn] skipping {path.name}: mode={spec.mode}, expected={expected_mode}")
            continue

        df = read_tsv(path)
        required = {"country", "check_country"}
        if expected_mode == "with_name" and group_by == "name":
            required.add("name")

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        df = df.copy()
        df["country"] = df["country"].fillna("").astype(str).str.strip()
        df["check_country"] = df["check_country"].fillna("").astype(str).str.strip()
        df = df[(df["country"] != "") & (df["country"] == df["check_country"])]

        if expected_mode == "with_name" and group_by == "name":
            df["name"] = df["name"].fillna("").astype(str).str.strip()
            df = df[df["name"] != ""]

        if df.empty:
            continue

        countries = df["country"].unique()
        if len(countries) != 1:
            raise ValueError(f"{path} contains more than one country: {countries.tolist()}")

        df["topic"] = spec.topic
        df["layer"] = spec.layer
        df["expansion"] = spec.expansion
        df["source_file"] = str(path)
        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def warn_unmatched_pairs(merged: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        preview = unmatched[keys + ["_merge"]].head(20).to_dict("records")
        print(
            f"[warn] dropping {len(unmatched)} rows without a with_name/woname "
            f"counterpart. First cases: {preview}"
        )
    return merged[merged["_merge"] == "both"].drop(columns="_merge")


def grouping_keys(group_by: str, include_topic: bool) -> tuple[list[str], list[str]]:
    baseline = ["layer", "expansion", "country"]
    with_name = baseline + (["name"] if group_by == "name" else [])
    if include_topic:
        baseline.append("topic")
        with_name.append("topic")
    return with_name, baseline


def build_default_bias_long(
    with_df: pd.DataFrame,
    without_df: pd.DataFrame,
    group_by: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_topic_keys, baseline_topic_keys = grouping_keys(group_by, include_topic=True)
    with_global_keys, baseline_global_keys = grouping_keys(group_by, include_topic=False)
    merge_how = "left" if group_by == "name" else "outer"
    validate = "many_to_one" if group_by == "name" else "one_to_one"

    with_topic = (
        with_df.groupby(with_topic_keys, as_index=False)
        .agg(ratio_with_name=("decision_binary", "mean"))
    )
    without_topic = (
        without_df.groupby(baseline_topic_keys, as_index=False)
        .agg(ratio_without_name=("decision_binary", "mean"))
    )
    by_topic = with_topic.merge(
        without_topic,
        on=baseline_topic_keys,
        how=merge_how,
        validate=validate,
        indicator=True,
    )
    by_topic = warn_unmatched_pairs(by_topic, with_topic_keys)
    by_topic["default_bias"] = by_topic["ratio_with_name"] - by_topic["ratio_without_name"]

    with_global = (
        with_df.groupby(with_global_keys, as_index=False)
        .agg(ratio_with_name=("decision_binary", "mean"))
    )
    without_global = (
        without_df.groupby(baseline_global_keys, as_index=False)
        .agg(ratio_without_name=("decision_binary", "mean"))
    )
    global_bias = with_global.merge(
        without_global,
        on=baseline_global_keys,
        how=merge_how,
        validate=validate,
        indicator=True,
    )
    global_bias = warn_unmatched_pairs(global_bias, with_global_keys)
    global_bias["default_bias"] = (
        global_bias["ratio_with_name"] - global_bias["ratio_without_name"]
    )

    return by_topic, global_bias[
        with_global_keys + ["ratio_with_name", "ratio_without_name", "default_bias"]
    ]


def ordered_topics(topics: Iterable[str]) -> list[str]:
    found = set(topics)
    preferred = [topic for topic in PREFERRED_TOPIC_ORDER if topic in found]
    return preferred + sorted(found - set(preferred))


def topic_label(topic: str) -> str:
    return topic.replace("_", " ").replace("-", " ").title()


def build_config_matrix(
    by_topic: pd.DataFrame,
    global_bias: pd.DataFrame,
    layer: str,
    expansion: str,
    group_by: str,
) -> pd.DataFrame:
    index_cols = ["country", "name"] if group_by == "name" else ["country"]
    topic_subset = by_topic[
        (by_topic["layer"] == layer) & (by_topic["expansion"] == expansion)
    ]
    global_subset = global_bias[
        (global_bias["layer"] == layer) & (global_bias["expansion"] == expansion)
    ].set_index(index_cols)

    topic_matrices = []
    for metric, suffix in {
        "ratio_with_name": "With Name",
        "ratio_without_name": "Without Name",
        "default_bias": None,
    }.items():
        metric_matrix = topic_subset.pivot(
            index=index_cols,
            columns="topic",
            values=metric,
        )
        metric_matrix = metric_matrix.rename(
            columns=lambda topic: (
                f"{topic_label(topic)} {suffix}" if suffix else topic_label(topic)
            )
        )
        topic_matrices.append(metric_matrix)

    matrix = pd.concat(topic_matrices, axis=1)
    matrix["Global With Name"] = global_subset["ratio_with_name"]
    matrix["Global Without Name"] = global_subset["ratio_without_name"]
    matrix["Global"] = global_subset["default_bias"]
    matrix = matrix.sort_index()

    if group_by == "name":
        matrix.loc[("Average", ""), :] = matrix.mean(axis=0, skipna=True)
    else:
        matrix.loc["Average"] = matrix.mean(axis=0, skipna=True)

    matrix = matrix.reset_index()
    matrix.columns.name = None

    ordered_columns = list(index_cols)
    for topic in ordered_topics(topic_subset["topic"].unique()):
        label = topic_label(topic)
        ordered_columns.extend([f"{label} With Name", f"{label} Without Name", label])
    ordered_columns.extend(["Global With Name", "Global Without Name", "Global"])
    return matrix[[column for column in ordered_columns if column in matrix.columns]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SAE-off default bias by country or by name."
    )
    parser.add_argument("--group-by", choices=["country", "name"], required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    root = ANALYSIS_DIR / JUDGE_MODEL
    with_dir = model_dir(root, "with_name", MODEL)
    without_dir = model_dir(root, "woname", MODEL)

    with_df = load_mode(with_dir, MODEL, "with_name", args.group_by)
    without_df = load_mode(without_dir, MODEL, "woname", args.group_by)

    print(f"[info] with_name rows: {len(with_df):,}")
    print(f"[info] woname rows: {len(without_df):,}")

    if with_df.empty:
        raise RuntimeError("No valid SAE-off rows found in with_name.")
    if without_df.empty:
        raise RuntimeError("No valid SAE-off rows found in woname.")

    by_topic, global_bias = build_default_bias_long(with_df, without_df, args.group_by)
    if by_topic.empty:
        raise RuntimeError("No compatible with_name/woname pairs found.")

    configs = sorted(
        set(zip(by_topic["layer"], by_topic["expansion"])),
        key=lambda value: (int(re.search(r"\d+", value[0]).group()), value[0], value[1]),
    )

    out_name = "default_bias_name" if args.group_by == "name" else "default_bias"
    out_dir = root / "output" / out_name / MODEL
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer, expansion in configs:
        out = build_config_matrix(by_topic, global_bias, layer, expansion, args.group_by)
        out_path = out_dir / f"{layer}_{expansion}.csv"
        out.to_csv(out_path, index=False)
        print(f"[saved] {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
