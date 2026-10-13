"""Compute default-bias summaries for regular judge outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
ENV = dotenv_values(ROOT / ".env")
DATA_DIR = Path(ENV["DATA_DIR"])

JUDGE_MODEL = ENV["JUDGE_MODEL_ID"]
ANALYSIS_ROOT = DATA_DIR / "analysis" / JUDGE_MODEL
MODEL = ENV["GENERATION_MODEL_SHORT"]
PREFERRED_TOPIC_ORDER = ("food", "clothing", "tradition", "rituals")


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


def model_dir(root: Path, name: str, model: str) -> Path:
    path = root / name / model
    if not path.exists():
        raise FileNotFoundError(f"Missing directory for {model}: {path}")
    return path


def topic_from_flat_file(path: Path) -> str | None:
    if "_FirstPerson_" not in path.stem:
        return None
    topic, _tail = path.stem.split("_FirstPerson_", 1)
    return topic


def parse_with_name_file(path: Path) -> tuple[str, str] | None:
    if "_" not in path.stem:
        return None
    country_a, country_b = path.stem.rsplit("_", 1)
    return country_a, country_b


def parse_without_name_file(path: Path) -> str | None:
    if not path.stem.startswith("baseline_"):
        return None
    return path.stem.removeprefix("baseline_")


def ordered_topics(with_dir: Path, without_dir: Path, group_by: str) -> list[str]:
    topics = {
        topic
        for base in (with_dir, without_dir)
        for path in base.glob("*.tsv")
        for topic in [topic_from_flat_file(path)]
        if topic is not None
    }
    if group_by == "country":
        topics.update(
            path.name
            for base in (with_dir, without_dir)
            for path in base.iterdir()
            if path.is_dir()
        )

    preferred = [topic for topic in PREFERRED_TOPIC_ORDER if topic in topics]
    return preferred + sorted(topics - set(preferred))


def paths_for_topic(base_dir: Path, topic: str, mode: str, group_by: str):
    topic_dir = base_dir / topic
    if group_by == "country" and topic_dir.exists():
        pattern = "*.tsv" if mode == "with_name" else "baseline_*.tsv"
        return topic_dir.glob(pattern)

    if mode == "with_name":
        return base_dir.glob(f"{topic}_FirstPerson_*.tsv")
    return base_dir.glob(f"{topic}_FirstPerson_baseline_*.tsv")


def country_from_with_name(path: Path, df: pd.DataFrame, group_by: str) -> str | None:
    if {"country", "check_country"}.issubset(df.columns):
        df["country"] = df["country"].astype(str).str.strip()
        df["check_country"] = df["check_country"].astype(str).str.strip()
        df.drop(df[df["country"] != df["check_country"]].index, inplace=True)
        if df.empty:
            return None

        countries = df["country"].dropna().unique()
        if len(countries) != 1:
            raise ValueError(f"{path} contains more than one country: {countries}")
        return str(countries[0])

    if group_by == "name":
        missing = {"country", "check_country"} - set(df.columns)
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    parsed = parse_with_name_file(path)
    if parsed is None:
        print(f"[warn] skipping malformed with_name file: {path}")
        return None

    country_a, country_b = parsed
    return country_a if country_a == country_b else None


def load_with_name(with_dir: Path, topics: list[str], group_by: str) -> pd.DataFrame:
    rows = []
    for topic in topics:
        for path in paths_for_topic(with_dir, topic, "with_name", group_by):
            df = read_tsv(path)
            df = df.copy()
            country = country_from_with_name(path, df, group_by)
            if country is None or df.empty:
                continue

            if group_by == "name":
                if "name" not in df.columns:
                    raise ValueError(f"{path} missing required column: name")
                df["name"] = df["name"].astype(str).str.strip()
                df = df[df["name"] != ""]
                if df.empty:
                    continue

            df["topic"] = topic
            df["country"] = country
            df["source_file"] = str(path)
            rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def country_from_without_name(path: Path, df: pd.DataFrame, group_by: str) -> str | None:
    if "check_country" in df.columns:
        countries = df["check_country"].astype(str).str.strip().dropna().unique()
        if len(countries) != 1:
            raise ValueError(f"{path} contains more than one check_country: {countries}")
        return str(countries[0])

    if group_by == "name":
        raise ValueError(f"{path} missing required column: check_country")

    country = parse_without_name_file(path)
    if country is None:
        print(f"[warn] skipping malformed woname file: {path}")
    return country


def load_without_name(without_dir: Path, topics: list[str], group_by: str) -> pd.DataFrame:
    rows = []
    for topic in topics:
        for path in paths_for_topic(without_dir, topic, "woname", group_by):
            df = read_tsv(path)
            country = country_from_without_name(path, df, group_by)
            if country is None:
                continue

            df = df.copy()
            df["topic"] = topic
            df["country"] = country
            df["source_file"] = str(path)
            rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def topic_label(topic: str) -> str:
    return topic.replace("_", " ").replace("-", " ").title()


def build_country_summary(with_df: pd.DataFrame, without_df: pd.DataFrame) -> pd.DataFrame:
    with_topic = (
        with_df.groupby(["country", "topic"], as_index=False)
        .agg(ratio_with_name=("decision_binary", "mean"))
    )
    without_topic = (
        without_df.groupby(["country", "topic"], as_index=False)
        .agg(ratio_without_name=("decision_binary", "mean"))
    )
    by_topic = with_topic.merge(
        without_topic,
        on=["country", "topic"],
        how="left",
        validate="one_to_one",
    )
    by_topic["default_bias"] = by_topic["ratio_with_name"] - by_topic["ratio_without_name"]

    matrices = []
    for metric, suffix in {
        "ratio_with_name": "With Name",
        "ratio_without_name": "Without Name",
        "default_bias": None,
    }.items():
        matrix = by_topic.pivot(index="country", columns="topic", values=metric)
        matrix = matrix.rename(
            columns=lambda topic: (
                f"{topic_label(topic)} {suffix}" if suffix else topic_label(topic)
            )
        )
        matrices.append(matrix)

    out = pd.concat(matrices, axis=1)

    with_global = (
        with_df.groupby("country", as_index=False)
        .agg(ratio_with_name=("decision_binary", "mean"))
    )
    without_global = (
        without_df.groupby("country", as_index=False)
        .agg(ratio_without_name=("decision_binary", "mean"))
    )
    global_bias = with_global.merge(
        without_global,
        on="country",
        how="left",
        validate="one_to_one",
    )
    global_bias["Global"] = (
        global_bias["ratio_with_name"] - global_bias["ratio_without_name"]
    )
    global_bias = global_bias.set_index("country")

    out["Global With Name"] = global_bias["ratio_with_name"]
    out["Global Without Name"] = global_bias["ratio_without_name"]
    out["Global"] = global_bias["Global"]
    out = out.sort_index()
    out.loc["Average"] = out.mean(axis=0, skipna=True)
    out = out.reset_index()
    out.columns.name = None
    return out


def build_name_summary(with_df: pd.DataFrame, without_df: pd.DataFrame) -> pd.DataFrame:
    with_topic = (
        with_df.groupby(["country", "name", "topic"], as_index=False)
        .agg(ratio_with_name=("decision_binary", "mean"))
    )
    without_topic = (
        without_df.groupby(["country", "topic"], as_index=False)
        .agg(ratio_without_name=("decision_binary", "mean"))
    )
    by_topic = with_topic.merge(
        without_topic,
        on=["country", "topic"],
        how="left",
        validate="many_to_one",
    )
    by_topic["default_bias"] = by_topic["ratio_with_name"] - by_topic["ratio_without_name"]

    out = by_topic.pivot(
        index=["country", "name"],
        columns="topic",
        values="default_bias",
    )

    with_global = (
        with_df.groupby(["country", "name"], as_index=False)
        .agg(ratio_with_name=("decision_binary", "mean"))
    )
    without_global = (
        without_df.groupby("country", as_index=False)
        .agg(ratio_without_name=("decision_binary", "mean"))
    )
    global_bias = with_global.merge(
        without_global,
        on="country",
        how="left",
        validate="many_to_one",
    )
    global_bias["Global"] = (
        global_bias["ratio_with_name"] - global_bias["ratio_without_name"]
    )

    out["Global"] = global_bias.set_index(["country", "name"])["Global"]
    out = out.sort_index().reset_index()
    out.columns.name = None
    return out


def order_columns(out: pd.DataFrame, topics: list[str], group_by: str) -> pd.DataFrame:
    if group_by == "name":
        out = out.rename(columns={topic: topic_label(topic) for topic in topics})
        topic_columns = [topic_label(topic) for topic in topics if topic_label(topic) in out.columns]
        columns = ["country", "name"] + topic_columns + ["Global"]
    else:
        columns = ["country"]
        for topic in topics:
            label = topic_label(topic)
            columns.extend([f"{label} With Name", f"{label} Without Name", label])
        columns.extend(["Global With Name", "Global Without Name", "Global"])

    return out[[column for column in columns if column in out.columns]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute default bias by country or by name."
    )
    parser.add_argument("--group-by", choices=["country", "name"], required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    root = ANALYSIS_ROOT
    with_dir = model_dir(root, "with_name", MODEL)
    without_dir = model_dir(root, "woname", MODEL)
    topics = ordered_topics(with_dir, without_dir, args.group_by)
    if not topics:
        raise RuntimeError(f"No topics/facets found in {with_dir} or {without_dir}")

    with_df = load_with_name(with_dir, topics, args.group_by)
    without_df = load_without_name(without_dir, topics, args.group_by)

    print(f"with_name rows: {len(with_df):,}")
    print(f"without_name rows: {len(without_df):,}")

    if with_df.empty:
        raise RuntimeError("No valid rows found in with_name.")
    if without_df.empty:
        raise RuntimeError("No valid rows found in woname.")

    if args.group_by == "name":
        out = build_name_summary(with_df, without_df)
        out_dir = root / "output" / "default_bias_name"
    else:
        out = build_country_summary(with_df, without_df)
        out_dir = root / "output" / "default_bias"

    out = order_columns(out, topics, args.group_by)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{MODEL}.csv"
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
