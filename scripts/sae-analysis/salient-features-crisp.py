"""Rank SAE features by country/facet and by topic using balanced bias labels.

Input parquets contain country, label, source_file, tokens, token_feature_idx,
and token_feature_act; token columns must align. Metadata supplies d_sae.
Country-level statistics balance biased and unbiased documents. Topic-level
statistics average country means with equal weights. Select TOP_N features
by ACD, RAR, and biased activation, prioritizing RAR >= TAU. Top tokens are
ranked by cumulative activation per biased document, with equal-country
averaging at topic level. Inputs are streamed in several Arrow passes."""

import argparse
from collections import defaultdict
import json
import os
import re
import sys
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from tqdm import tqdm
from crisp_utils import (
    GroupAggregate,
    compute_country_salience,
    compute_macro_topic_salience,
    infer_topic_from_source_file,
    iter_token_activations,
    normalize_token,
    rank_tokens,
    save_output,
    write_metadata_json,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ENV = dotenv_values(os.path.join(ROOT, ".env"))
DATA_DIR = Path(ENV["DATA_DIR"])


META_COLUMNS = ["country", "label", "source_file"]
ACTIVATION_COLUMNS = META_COLUMNS + ["token_feature_idx", "token_feature_act"]
TOKEN_COLUMNS = ACTIVATION_COLUMNS + ["tokens"]

JUDGE = ENV["JUDGE_MODEL_ID"]
MODEL = ENV["GENERATION_MODEL_SHORT"]
SITE = ENV["SITE"]
EXPANSION = ENV["EXPANSION"]
TOP_N = int(ENV["TOP_N"])
TOP_TOKENS = int(ENV["TOP_TOKENS"])
TAU = float(ENV["TAU"])
OUTPUT_DIR = str(DATA_DIR / "crisp" / JUDGE)
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


def input_file_for_layer(layer: int, input_dir=None) -> str:
    return str(Path(input_dir or DATA_DIR / "sae" / JUDGE) / f"{MODEL}_L{layer}{SITE}_{EXPANSION}.parquet")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CRISP-style salient SAE features from one token-level activation parquet."
    )
    parser.add_argument("--input_dir", type=Path, default=DATA_DIR / "sae" / JUDGE)
    parser.add_argument("--out_dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--countries", nargs="+", default=TARGET_COUNTRIES)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of parquet rows to process per PyArrow batch.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Small constant added to the RAR denominator.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed for label balancing.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero.")
    if TOP_N <= 0:
        parser.error("TOP_N must be greater than zero.")
    if TOP_TOKENS <= 0:
        parser.error("TOP_TOKENS must be greater than zero.")
    if TAU < 0:
        parser.error("TAU cannot be negative.")
    if args.epsilon <= 0:
        parser.error("--epsilon must be greater than zero.")

    return args


def resolve_input_file(input_file: str) -> str:
    input_file = os.path.abspath(input_file)
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input parquet not found: {input_file}")
    if not input_file.lower().endswith(".parquet"):
        raise ValueError(f"Input file must have a .parquet extension: {input_file}")
    return input_file


def metadata_path_for_input(input_file: str) -> str:
    return input_file + ".meta.json"


def load_d_sae(input_file: str) -> Tuple[int, str]:
    meta_path = metadata_path_for_input(input_file)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Extractor metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(f"Input metadata must contain a JSON object: {meta_path}")
    if "d_sae" not in metadata:
        raise ValueError(f"Missing required field 'd_sae' in {meta_path}")

    d_sae = int(metadata["d_sae"])
    if d_sae <= 0:
        raise ValueError("Metadata field 'd_sae' must be greater than zero.")

    return d_sae, meta_path


def paths_for_run(input_file: str) -> Tuple[str, str]:
    out_dir = os.path.abspath(OUTPUT_DIR)
    input_stem = os.path.splitext(os.path.basename(input_file))[0]
    run_id = input_stem.removeprefix("llamascope_crisp_")

    out_topic = os.path.join(
        out_dir,
        f"topic_salient_features_{run_id}.parquet",
    )
    out_topic_country = os.path.join(
        out_dir,
        f"topic_country_salient_features_{run_id}.parquet",
    )

    return out_topic, out_topic_country

def parquet_schema_columns(path: str) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to inspect and stream the activation parquet."
        ) from exc

    return pq.ParquetFile(path).schema_arrow.names


def iter_parquet_batches(
    path: str,
    columns: Sequence[str],
    batch_size: int,
    desc: str,
) -> Iterator[pd.DataFrame]:
    """Stream selected parquet columns as pandas batches."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to stream the activation parquet."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = set(columns) - available
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}. "
            f"Expected token-level schema columns: {TOKEN_COLUMNS}."
        )

    total_rows = parquet_file.metadata.num_rows
    with tqdm(total=total_rows, desc=desc, unit="doc") as progress:
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(columns),
            use_threads=True,
        ):
            df = record_batch.to_pandas()
            progress.update(len(df))
            yield df


def collect_balanced_doc_ids(
    input_file: str,
    random_state: int,
    batch_size: int,
) -> Tuple[Set[int], Dict[str, int]]:
    """Balance labels within every (topic, country) stratum."""
    strata: DefaultDict[Tuple[str, str, int], List[int]] = defaultdict(list)
    doc_offset = 0

    for df in iter_parquet_batches(
        input_file,
        columns=META_COLUMNS,
        batch_size=batch_size,
        desc="reading metadata",
    ):
        labels = df["label"].astype(int)
        invalid_labels = sorted(set(labels.unique()) - {0, 1})
        if invalid_labels:
            raise ValueError(f"Expected binary labels 0/1, found: {invalid_labels}")

        topics = df["source_file"].apply(infer_topic_from_source_file)

        for local_idx, (topic, country, label) in enumerate(
            zip(topics, df["country"], labels)
        ):
            if (
                TARGET_COUNTRY_KEYS
                and str(country).strip().casefold() not in TARGET_COUNTRY_KEYS
            ):
                continue
            strata[(str(topic), str(country), int(label))].append(
                doc_offset + local_idx
            )

        doc_offset += len(df)

    rng = np.random.default_rng(random_state)
    selected_ids: Set[int] = set()
    n_groups_total = 0
    n_groups_selected = 0
    n_groups_dropped = 0
    n_selected_per_label = 0

    topic_country_keys = sorted(
        {(topic, country) for topic, country, _ in strata},
        key=lambda item: (item[0], item[1]),
    )

    for topic, country in topic_country_keys:
        n_groups_total += 1
        ids0 = strata.get((topic, country, 0), [])
        ids1 = strata.get((topic, country, 1), [])
        n = min(len(ids0), len(ids1))

        if n == 0:
            n_groups_dropped += 1
            continue

        selected_0 = rng.choice(ids0, size=n, replace=False)
        selected_1 = rng.choice(ids1, size=n, replace=False)
        selected_ids.update(int(doc_id) for doc_id in selected_0)
        selected_ids.update(int(doc_id) for doc_id in selected_1)

        n_groups_selected += 1
        n_selected_per_label += n

    diagnostics = {
        "n_topic_country_groups_total": n_groups_total,
        "n_topic_country_groups_selected": n_groups_selected,
        "n_topic_country_groups_dropped": n_groups_dropped,
        "n_selected_label_0": n_selected_per_label,
        "n_selected_label_1": n_selected_per_label,
        "n_selected_total": 2 * n_selected_per_label,
    }

    return selected_ids, diagnostics


def selected_local_indices(
    n_rows: int,
    doc_offset: int,
    selected_ids: Set[int],
) -> List[int]:
    return [
        local_idx
        for local_idx in range(n_rows)
        if doc_offset + local_idx in selected_ids
    ]


def aggregate_selected_documents(
    input_file: str,
    selected_ids: Set[int],
    d_sae: int,
    batch_size: int,
) -> Dict[Tuple[str, str], GroupAggregate]:
    """Aggregate selected documents separately for every (topic, country)."""
    topic_country_aggs: Dict[Tuple[str, str], GroupAggregate] = {}
    doc_offset = 0

    for df in iter_parquet_batches(
        input_file,
        columns=ACTIVATION_COLUMNS,
        batch_size=batch_size,
        desc="aggregating selected docs",
    ):
        n_rows = len(df)
        local_indices = selected_local_indices(n_rows, doc_offset, selected_ids)

        if local_indices:
            selected_df = df.iloc[local_indices].copy()
            selected_df["label"] = selected_df["label"].astype(int)
            selected_df["topic"] = selected_df["source_file"].apply(
                infer_topic_from_source_file
            )

            for row in selected_df.itertuples(index=False):
                key = (str(row.topic), str(row.country))
                aggregate = topic_country_aggs.setdefault(
                    key,
                    GroupAggregate(d_sae),
                )
                aggregate.add_document(
                    label=int(row.label),
                    token_feature_idx=row.token_feature_idx,
                    token_feature_act=row.token_feature_act,
                )

        doc_offset += n_rows

    return topic_country_aggs


def select_salient_features(
    salience_df: pd.DataFrame,
    top_n: int,
    tau: float,
) -> List[int]:
    """
    Prioritize RAR >= tau, ranking by ACD, RAR, and biased activation.

    If fewer than top_n features pass tau, fill the remaining positions using
    the same ranking over the other features. This preserves a fixed-width
    top-N output while keeping threshold-passing features first.
    """
    if salience_df.empty:
        return []

    ranked = salience_df.sort_values(
        ["acd", "rar", "cum_act_biased", "feature"],
        ascending=[False, False, False, True],
    )
    strong = ranked[ranked["rar"] >= tau]
    features = strong["feature"].astype(int).tolist()

    if len(features) < top_n:
        remaining = ranked[~ranked["feature"].isin(features)]["feature"]
        features.extend(
            remaining.astype(int).tolist()[: top_n - len(features)]
        )

    return features[:top_n]


def metric_vectors_for_features(
    salience_df: pd.DataFrame,
    features: Sequence[int],
) -> Tuple[List[float], List[float], List[float]]:
    if not features:
        return [], [], []

    selected = salience_df.set_index("feature").loc[list(features)]
    acd = selected["acd"].astype(float).tolist()
    rar = selected["rar"].astype(float).tolist()
    act_sum = (
        selected["cum_act_biased"].to_numpy(dtype=np.float64)
        + selected["cum_act_unbiased"].to_numpy(dtype=np.float64)
    ).astype(float).tolist()
    return acd, rar, act_sum


def build_topic_rows_macro(
    topic_country_aggs: Dict[Tuple[str, str], GroupAggregate],
    epsilon: float,
    top_n: int,
    tau: float,
) -> pd.DataFrame:
    rows = []
    topics = sorted({topic for topic, _country in topic_country_aggs})

    for topic in tqdm(topics, desc="macro topic groups", unit="topic"):
        salience_df, n1, n0, nt, n_countries = compute_macro_topic_salience(
            topic=topic,
            topic_country_aggs=topic_country_aggs,
            epsilon=epsilon,
        )
        if salience_df.empty:
            continue

        features = select_salient_features(salience_df, top_n=top_n, tau=tau)
        acd, rar, act_sum = metric_vectors_for_features(salience_df, features)
        rows.append(
            {
                "topic": topic,
                "salient_features": features,
                "salient_acd": acd,
                "salient_rar": rar,
                "salient_act_sum": act_sum,
                "n_countries": n_countries,
                "n_docs_biased": n1,
                "n_docs_unbiased": n0,
                "n_docs_total": nt,
            }
        )

    return pd.DataFrame(rows)


def build_topic_country_rows(
    topic_country_aggs: Dict[Tuple[str, str], GroupAggregate],
    epsilon: float,
    top_n: int,
    tau: float,
) -> pd.DataFrame:
    rows = []
    groups = sorted(topic_country_aggs, key=lambda item: (item[0], item[1]))

    for topic, country in tqdm(
        groups,
        desc="topic-country groups",
        unit="group",
    ):
        salience_df, n1, n0, nt = compute_country_salience(
            aggregate=topic_country_aggs[(topic, country)],
            epsilon=epsilon,
        )
        if salience_df.empty:
            continue

        features = select_salient_features(salience_df, top_n=top_n, tau=tau)
        acd, rar, act_sum = metric_vectors_for_features(salience_df, features)
        rows.append(
            {
                "topic": topic,
                "country": country,
                "salient_features": features,
                "salient_acd": acd,
                "salient_rar": rar,
                "salient_act_sum": act_sum,
                "n_docs_biased": n1,
                "n_docs_unbiased": n0,
                "n_docs_total": nt,
            }
        )

    return pd.DataFrame(rows)


def selected_feature_sets(
    df_topic: pd.DataFrame,
    df_topic_country: pd.DataFrame,
) -> Tuple[Dict[str, Set[int]], Dict[Tuple[str, str], Set[int]]]:
    topic_features = {
        str(row.topic): {int(feature) for feature in row.salient_features}
        for row in df_topic.itertuples(index=False)
    }
    topic_country_features = {
        (str(row.topic), str(row.country)): {
            int(feature) for feature in row.salient_features
        }
        for row in df_topic_country.itertuples(index=False)
    }
    return topic_features, topic_country_features


def collect_biased_token_scores(
    input_file: str,
    selected_ids: Set[int],
    topic_features: Dict[str, Set[int]],
    topic_country_features: Dict[Tuple[str, str], Set[int]],
    batch_size: int,
) -> Tuple[
    Dict[Tuple[str, str], Dict[int, Dict[str, float]]],
    Dict[Tuple[str, str], int],
]:
    """Collect cumulative token activations for selected features in label=1."""
    token_scores: DefaultDict[
        Tuple[str, str],
        DefaultDict[int, DefaultDict[str, float]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    n_biased_docs: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    doc_offset = 0

    for df in iter_parquet_batches(
        input_file,
        columns=TOKEN_COLUMNS,
        batch_size=batch_size,
        desc="collecting top tokens",
    ):
        n_rows = len(df)
        local_indices = selected_local_indices(n_rows, doc_offset, selected_ids)

        if local_indices:
            selected_df = df.iloc[local_indices].copy()
            selected_df["label"] = selected_df["label"].astype(int)
            selected_df["topic"] = selected_df["source_file"].apply(
                infer_topic_from_source_file
            )

            for row in selected_df.itertuples(index=False):
                if int(row.label) != 1:
                    continue

                key = (str(row.topic), str(row.country))
                n_biased_docs[key] += 1
                target_features = topic_features.get(key[0], set()) | (
                    topic_country_features.get(key, set())
                )
                if not target_features:
                    continue

                tokens = row.tokens
                if tokens is None:
                    raise ValueError(f"tokens cannot be null for group {key}.")
                if len(tokens) != len(row.token_feature_idx):
                    raise ValueError(
                        f"tokens/token_feature_idx position mismatch for group {key}: "
                        f"{len(tokens)} vs {len(row.token_feature_idx)}."
                    )

                activation_rows = iter_token_activations(
                    row.token_feature_idx,
                    row.token_feature_act,
                )
                for token, (feature_ids, feature_acts) in zip(tokens, activation_rows):
                    token_string = normalize_token(token)
                    for feature, activation in zip(feature_ids, feature_acts):
                        feature_id = int(feature)
                        if feature_id in target_features:
                            token_scores[key][feature_id][token_string] += float(
                                activation
                            )

        doc_offset += n_rows

    plain_scores = {
        key: {
            feature: dict(scores)
            for feature, scores in feature_map.items()
        }
        for key, feature_map in token_scores.items()
    }
    return plain_scores, dict(n_biased_docs)


def topic_country_top_token_matrix(
    topic: str,
    country: str,
    features: Sequence[int],
    token_scores: Dict[Tuple[str, str], Dict[int, Dict[str, float]]],
    n_biased_docs: Dict[Tuple[str, str], int],
    top_k: int,
) -> List[List[Optional[str]]]:
    key = (str(topic), str(country))
    denominator = n_biased_docs.get(key, 0)
    matrix: List[List[Optional[str]]] = []

    for feature in features:
        if denominator == 0:
            matrix.append([])
            continue

        scores = token_scores.get(key, {}).get(int(feature), {})
        mean_scores = {
            token: activation / denominator
            for token, activation in scores.items()
        }
        matrix.append(rank_tokens(mean_scores, top_k=top_k))

    return matrix


def topic_top_token_matrix(
    topic: str,
    features: Sequence[int],
    topic_country_aggs: Dict[Tuple[str, str], GroupAggregate],
    token_scores: Dict[Tuple[str, str], Dict[int, Dict[str, float]]],
    n_biased_docs: Dict[Tuple[str, str], int],
    top_k: int,
) -> List[List[Optional[str]]]:
    """Macro-average per-document token activation scores across countries."""
    country_keys = sorted(
        [
            key
            for key, aggregate in topic_country_aggs.items()
            if key[0] == topic
            and aggregate.n_docs[0] > 0
            and aggregate.n_docs[1] > 0
            and n_biased_docs.get(key, 0) > 0
        ],
        key=lambda key: key[1],
    )

    matrix: List[List[Optional[str]]] = []
    for feature in features:
        macro_scores: DefaultDict[str, float] = defaultdict(float)

        for key in country_keys:
            denominator = n_biased_docs[key]
            country_scores = token_scores.get(key, {}).get(int(feature), {})
            for token, activation in country_scores.items():
                macro_scores[token] += activation / denominator

        if country_keys:
            n_countries = len(country_keys)
            normalized_scores = {
                token: score / n_countries
                for token, score in macro_scores.items()
            }
        else:
            normalized_scores = {}

        matrix.append(rank_tokens(normalized_scores, top_k=top_k))

    return matrix


def add_top_token_columns(
    df_topic: pd.DataFrame,
    df_topic_country: pd.DataFrame,
    input_file: str,
    selected_ids: Set[int],
    topic_country_aggs: Dict[Tuple[str, str], GroupAggregate],
    top_k: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    topic_features, topic_country_features = selected_feature_sets(
        df_topic,
        df_topic_country,
    )
    token_scores, n_biased_docs = collect_biased_token_scores(
        input_file=input_file,
        selected_ids=selected_ids,
        topic_features=topic_features,
        topic_country_features=topic_country_features,
        batch_size=batch_size,
    )

    df_topic = df_topic.copy()
    df_topic_country = df_topic_country.copy()

    df_topic["salient_top_tokens"] = [
        topic_top_token_matrix(
            topic=str(row.topic),
            features=row.salient_features,
            topic_country_aggs=topic_country_aggs,
            token_scores=token_scores,
            n_biased_docs=n_biased_docs,
            top_k=top_k,
        )
        for row in df_topic.itertuples(index=False)
    ]

    df_topic_country["salient_top_tokens"] = [
        topic_country_top_token_matrix(
            topic=str(row.topic),
            country=str(row.country),
            features=row.salient_features,
            token_scores=token_scores,
            n_biased_docs=n_biased_docs,
            top_k=top_k,
        )
        for row in df_topic_country.itertuples(index=False)
    ]

    return df_topic, df_topic_country


def process_input_file(args, input_file: str) -> None:
    input_file = resolve_input_file(input_file)
    d_sae, input_meta_path = load_d_sae(input_file)
    out_topic, out_topic_country = paths_for_run(input_file)

    print(f"d_sae: {d_sae}")

    # Validate the full required schema before performing the expensive passes.
    available_columns = set(parquet_schema_columns(input_file))
    missing_columns = set(TOKEN_COLUMNS) - available_columns
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {input_file}: {sorted(missing_columns)}. "
            f"Expected: {TOKEN_COLUMNS}."
        )

    selected_ids, balance_diagnostics = collect_balanced_doc_ids(
        input_file=input_file,
        random_state=args.random_state,
        batch_size=args.batch_size,
    )
    if not selected_ids:
        raise RuntimeError(
            "Balanced subset is empty. Check label distribution within each "
            "(topic, country)."
        )

    topic_country_aggs = aggregate_selected_documents(
        input_file=input_file,
        selected_ids=selected_ids,
        d_sae=d_sae,
        batch_size=args.batch_size,
    )
    if not topic_country_aggs:
        raise RuntimeError("No selected documents were aggregated.")

    df_topic = build_topic_rows_macro(
        topic_country_aggs=topic_country_aggs,
        epsilon=args.epsilon,
        top_n=TOP_N,
        tau=TAU,
    )
    df_topic_country = build_topic_country_rows(
        topic_country_aggs=topic_country_aggs,
        epsilon=args.epsilon,
        top_n=TOP_N,
        tau=TAU,
    )

    if df_topic.empty:
        raise RuntimeError("Topic-level output is empty.")
    if df_topic_country.empty:
        raise RuntimeError("Topic-country-level output is empty.")

    df_topic, df_topic_country = add_top_token_columns(
        df_topic=df_topic,
        df_topic_country=df_topic_country,
        input_file=input_file,
        selected_ids=selected_ids,
        topic_country_aggs=topic_country_aggs,
        top_k=TOP_TOKENS,
        batch_size=args.batch_size,
    )

    save_output(df_topic, out_topic)
    save_output(df_topic_country, out_topic_country)

    common_metadata = {
        "input_file": input_file,
        "input_metadata_file": input_meta_path,
        "out_topic": out_topic,
        "out_topic_country": out_topic_country,
        "input_processing": "single parquet streamed with pyarrow batches",
        "batch_size": int(args.batch_size),
        "required_input_columns": TOKEN_COLUMNS,
        "d_sae": int(d_sae),
        "top_n": int(TOP_N),
        "top_tokens": int(TOP_TOKENS),
        "tau": float(TAU),
        "epsilon": float(args.epsilon),
        "random_state": int(args.random_state),
        "target_countries": TARGET_COUNTRIES,
        "selection": {
            "method": "balance_label_within_topic_country",
            **balance_diagnostics,
        },
        "feature_selection": {
            "priority_filter": "rar >= tau",
            "ranking": ["acd descending", "rar descending", "cum_act_biased descending"],
            "fill_policy": "fill to top_n from remaining ranked features",
        },
        "salience": {
            "acd": "count_biased - count_unbiased",
            "rar": "cum_act_biased / (cum_act_unbiased + epsilon)",
            "act_sum": "cum_act_biased + cum_act_unbiased",
        },
        "top_token_matrix": {
            "column": "salient_top_tokens",
            "source_label": 1,
            "alignment": "salient_top_tokens[i] corresponds to salient_features[i]; each row has exactly top_tokens entries and is padded with nulls",
            "topic_country_score": "cumulative feature activation per selected biased document",
            "topic_score": "equal-country macro-average of cumulative feature activation per selected biased document",
        },
        "columns_aligned_with_salient_features": [
            "salient_acd",
            "salient_rar",
            "salient_act_sum",
            "salient_top_tokens",
        ],
    }

    topic_meta_path = write_metadata_json(
        out_topic,
        {
            **common_metadata,
            "grouping": "topic",
            "country_weighting": (
                "equal-country macro-average after computing per-document "
                "feature means within each country and label"
            ),
        },
    )
    topic_country_meta_path = write_metadata_json(
        out_topic_country,
        {
            **common_metadata,
            "grouping": "topic_country",
            "country_weighting": "not applicable",
        },
    )

    print(f"Saved topic-level parquet: {out_topic}")
    print(f"Saved topic-level metadata: {topic_meta_path}")
    print(f"Saved topic-country-level parquet: {out_topic_country}")
    print(f"Saved topic-country-level metadata: {topic_country_meta_path}")


def main():
    global OUTPUT_DIR, TARGET_COUNTRIES, TARGET_COUNTRY_KEYS
    args = parse_args()
    OUTPUT_DIR = str(args.out_dir)
    TARGET_COUNTRIES = args.countries
    TARGET_COUNTRY_KEYS = {country.casefold() for country in TARGET_COUNTRIES}
    inputs = [input_file_for_layer(layer, args.input_dir) for layer in LAYERS]
    for input_file in inputs:
        for output_file in paths_for_run(input_file):
            for candidate in (output_file, output_file + ".meta.json"):
                if os.path.exists(candidate):
                    raise FileExistsError(f"Output already exists: {candidate}. Choose another --out_dir.")
    for input_file in inputs:
        process_input_file(args, input_file)


if __name__ == "__main__":
    main()
