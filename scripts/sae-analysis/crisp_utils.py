"""Shared helpers for CRISP salient-feature scripts."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PERSON_MARKERS = {"FirstPerson", "SecondPerson", "ThirdPerson"}


def is_nonempty_sequence(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0


def iter_token_activations(token_feature_idx, token_feature_act) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if token_feature_idx is None or token_feature_act is None:
        raise ValueError("token_feature_idx and token_feature_act cannot be null.")
    if len(token_feature_idx) != len(token_feature_act):
        raise ValueError(
            "token_feature_idx/token_feature_act position mismatch: "
            f"{len(token_feature_idx)} vs {len(token_feature_act)}."
        )

    for feature_ids, feature_acts in zip(token_feature_idx, token_feature_act):
        if not is_nonempty_sequence(feature_ids):
            if is_nonempty_sequence(feature_acts):
                raise ValueError("A token has activation values but no feature ids.")
            yield np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
            continue

        ids = np.asarray(feature_ids, dtype=np.int64).ravel()
        acts = np.asarray(feature_acts, dtype=np.float64).ravel()
        if len(ids) != len(acts):
            raise ValueError(f"Per-token feature/activation mismatch: {len(ids)} vs {len(acts)}.")
        if not np.all(np.isfinite(acts)):
            raise ValueError("Found a non-finite SAE activation.")
        yield ids, acts


class GroupAggregate:
    def __init__(self, feature_space: int):
        self.count = {
            0: np.zeros(feature_space, dtype=np.float64),
            1: np.zeros(feature_space, dtype=np.float64),
        }
        self.sum = {
            0: np.zeros(feature_space, dtype=np.float64),
            1: np.zeros(feature_space, dtype=np.float64),
        }
        self.n_docs = {0: 0, 1: 0}

    def add_document(self, label: int, token_feature_idx, token_feature_act) -> None:
        if label not in (0, 1):
            raise ValueError(f"Expected label 0 or 1, got {label}.")

        self.n_docs[label] += 1
        for feature_ids, feature_acts in iter_token_activations(
            token_feature_idx,
            token_feature_act,
        ):
            if feature_ids.size == 0:
                continue
            if feature_ids.min() < 0 or feature_ids.max() >= len(self.count[label]):
                raise ValueError(
                    "Feature id outside the configured feature space: "
                    f"min={feature_ids.min()}, max={feature_ids.max()}, "
                    f"feature_space={len(self.count[label])}."
                )
            np.add.at(self.count[label], feature_ids, 1.0)
            np.add.at(self.sum[label], feature_ids, feature_acts)


def infer_topic_from_source_file(source_file: str) -> str:
    stem = os.path.splitext(os.path.basename(str(source_file)))[0]
    parts = stem.split("_")
    if len(parts) < 2 or parts[1] not in PERSON_MARKERS:
        raise ValueError(
            "Cannot infer topic from source_file. Expected topic_<Person>_..., "
            f"got: {source_file}"
        )
    return parts[0]


def salience_dataframe(
    count_unbiased: np.ndarray,
    count_biased: np.ndarray,
    sum_unbiased: np.ndarray,
    sum_biased: np.ndarray,
    epsilon: float,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "feature": np.arange(len(count_unbiased), dtype=np.int64),
            "count_biased": count_biased,
            "count_unbiased": count_unbiased,
            "cum_act_biased": sum_biased,
            "cum_act_unbiased": sum_unbiased,
        }
    )
    out["acd"] = out["count_biased"] - out["count_unbiased"]
    out["rar"] = out["cum_act_biased"] / (out["cum_act_unbiased"] + epsilon)
    return out


def compute_country_salience(
    aggregate: GroupAggregate,
    epsilon: float,
) -> Tuple[pd.DataFrame, int, int, int]:
    n0 = aggregate.n_docs[0]
    n1 = aggregate.n_docs[1]
    if n0 == 0 or n1 == 0:
        return pd.DataFrame(), n1, n0, n0 + n1

    return (
        salience_dataframe(
            count_unbiased=aggregate.count[0],
            count_biased=aggregate.count[1],
            sum_unbiased=aggregate.sum[0],
            sum_biased=aggregate.sum[1],
            epsilon=epsilon,
        ),
        n1,
        n0,
        n0 + n1,
    )


def compute_macro_topic_salience(
    topic: str,
    topic_country_aggs: Dict[Tuple[str, str], GroupAggregate],
    epsilon: float,
) -> Tuple[pd.DataFrame, int, int, int, int]:
    country_aggs = [
        aggregate
        for (group_topic, _country), aggregate in topic_country_aggs.items()
        if group_topic == topic
        and aggregate.n_docs[0] > 0
        and aggregate.n_docs[1] > 0
    ]
    if not country_aggs:
        return pd.DataFrame(), 0, 0, 0, 0

    mean_count_0 = np.stack(
        [aggregate.count[0] / aggregate.n_docs[0] for aggregate in country_aggs]
    ).mean(axis=0)
    mean_count_1 = np.stack(
        [aggregate.count[1] / aggregate.n_docs[1] for aggregate in country_aggs]
    ).mean(axis=0)
    mean_sum_0 = np.stack(
        [aggregate.sum[0] / aggregate.n_docs[0] for aggregate in country_aggs]
    ).mean(axis=0)
    mean_sum_1 = np.stack(
        [aggregate.sum[1] / aggregate.n_docs[1] for aggregate in country_aggs]
    ).mean(axis=0)
    n0 = sum(aggregate.n_docs[0] for aggregate in country_aggs)
    n1 = sum(aggregate.n_docs[1] for aggregate in country_aggs)

    return (
        salience_dataframe(mean_count_0, mean_count_1, mean_sum_0, mean_sum_1, epsilon),
        n1,
        n0,
        n0 + n1,
        len(country_aggs),
    )


def normalize_token(token) -> str:
    if isinstance(token, bytes):
        return token.decode("utf-8", errors="replace")
    return str(token)


def rank_tokens(token_scores: Dict[str, float], top_k: int) -> List[Optional[str]]:
    ranked = sorted(token_scores.items(), key=lambda item: (-item[1], item[0]))
    tokens: List[Optional[str]] = [token for token, _score in ranked[:top_k]]
    tokens.extend([None] * (top_k - len(tokens)))
    return tokens


def save_output(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def write_metadata_json(path: str, metadata: Dict) -> str:
    meta_path = path + ".meta.json"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return meta_path
