"""Shared helpers for response generation scripts."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pycountry
import torch

from external import open_ended_qa_cleaned as Q
from external import prompts as P


SUPPORTED_TOPICS = ("food", "drink", "tradition", "clothing", "rituals")


@dataclass(frozen=True)
class LayerSpec:
    layer: int
    site: str
    layer_id: str


def unique_preserving_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def slugify(value: str) -> str:
    cleaned = str(value).strip().lower().replace("/", "_")
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", cleaned)
    return cleaned.strip("_")


def normalize_key(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


def parse_layer_spec(raw: str, default_site: str) -> LayerSpec:
    match = re.fullmatch(r"L?(\d+)(TC|R|A|M)?", str(raw).strip().upper())
    if not match:
        raise ValueError(f"Invalid layer identifier {raw!r}.")

    layer = int(match.group(1))
    site = match.group(2) or default_site
    if site != "R":
        raise NotImplementedError(
            f"Layer {raw!r} uses site {site}. Only residual-stream site R is supported."
        )
    return LayerSpec(layer=layer, site=site, layer_id=f"L{layer}{site}")


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def load_fb_names_json(path: Path, max_names_per_country: int) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    output: Dict[str, List[str]] = {}
    male = data.get("M", {}) or {}
    female = data.get("F", {}) or {}

    for alpha2 in sorted(set(male) | set(female)):
        country_obj = pycountry.countries.get(alpha_2=str(alpha2))
        if country_obj is None:
            continue

        names: List[str] = []
        names.extend((male.get(alpha2) or [])[:max_names_per_country])
        names.extend((female.get(alpha2) or [])[:max_names_per_country])
        unique_names = unique_preserving_order(names)
        if unique_names:
            output[country_obj.name] = unique_names

    return output


def resolve_country_name(requested_country: str, names_by_country: Dict[str, List[str]]) -> str:
    if requested_country in names_by_country:
        return requested_country

    normalized = normalize_key(requested_country)
    matches = [
        country for country in names_by_country
        if normalize_key(country) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Country name {requested_country!r} is ambiguous: {matches}")
    raise ValueError(f"Country {requested_country!r} was not found in the names file.")






def get_questions(topic: str) -> List[str]:
    mapping = {
        "food": "food_questions_first_third_person",
        "drink": "food_questions_first_third_person",
        "tradition": "tradition_questions_first_third_person",
        "clothing": "clothing_questions_first_third_person",
        "rituals": "ritual_questions_first_third_person",
    }
    questions = getattr(Q, mapping[topic])
    return [question.strip() for question in questions if question and question.strip()]


def build_system_prompt(
    mode: str,
    person: str,
    name: Optional[str],
    country: Optional[str],
) -> str:
    if mode == "with_name":
        return P.system_prompts_name[person].format(name)
    if mode == "with_country":
        return P.system_prompts_country[person].format(country)
    if mode == "name_and_country":
        return P.system_prompts_name_country[person].format(name, country)
    if mode == "woname":
        return P.system_prompts_wonames[person]
    raise ValueError(f"Unsupported mode: {mode}")


def chat_text(tokenizer, system_text: str, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_tasks(
    mode: str,
    person: str,
    topic: str,
    country: str,
    names_by_country: Dict[str, List[str]],
) -> List[Tuple[str, Optional[str], str, str]]:
    questions = get_questions(topic)
    tasks: List[Tuple[str, Optional[str], str, str]] = []

    if mode == "woname":
        system_prompt = build_system_prompt(mode, person, None, None)
        return [(country, None, question, system_prompt) for question in questions]

    if mode == "with_country":
        system_prompt = build_system_prompt(mode, person, None, country)
        return [(country, None, question, system_prompt) for question in questions]

    names = names_by_country.get(country, [])
    if not names:
        raise ValueError(f"No names available for country {country!r}.")

    if mode == "with_name":
        for name in names:
            system_prompt = build_system_prompt(mode, person, name, None)
            tasks.extend((country, name, question, system_prompt) for question in questions)
        return tasks

    if mode == "name_and_country":
        for name in names:
            system_prompt = build_system_prompt(mode, person, name, country)
            tasks.extend((country, name, question, system_prompt) for question in questions)
        return tasks

    raise ValueError(f"Unsupported mode: {mode}")


def parse_feature_ids(value, *, deduplicate: bool = True) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        value = ast.literal_eval(stripped)
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        features = [int(feature) for feature in value if feature is not None]
    elif pd.isna(value):
        return []
    else:
        features = [int(value)]

    return unique_preserving_order(features) if deduplicate else features


def load_layer_feature_rows(
    path: Path,
    country: str,
    topics: Sequence[str],
    num_features: Optional[int],
    *,
    skip_missing: bool = False,
) -> Dict[str, Tuple[int, ...]]:
    """Load features, optionally omitting absent country/topic rows and empty lists.

    Missing files, duplicate rows, and malformed feature values remain errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Salient-feature parquet not found: {path}.")

    required_columns = ["topic", "country", "salient_features"]
    df = pd.read_parquet(path, columns=required_columns)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    country_df = df[
        df["country"].astype(str).map(normalize_key) == normalize_key(country)
    ].copy()
    if country_df.empty:
        if skip_missing:
            print(f"[skip] No salient-feature rows for country={country!r} in {path}.")
            return {}
        available = sorted(df["country"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"Country {country!r} was not found in {path}. "
            f"First available countries: {available[:20]}"
        )

    country_df["topic_normalized"] = (
        country_df["topic"].astype(str).str.strip().str.lower()
    )
    output: Dict[str, Tuple[int, ...]] = {}
    for topic in topics:
        rows = country_df[country_df["topic_normalized"] == topic]
        if rows.empty:
            if skip_missing:
                print(
                    f"[skip] No salient-feature row for topic={topic!r}, "
                    f"country={country!r} in {path}."
                )
                continue
            available_topics = sorted(country_df["topic_normalized"].dropna().unique().tolist())
            raise ValueError(
                f"No salient-feature row for topic={topic!r}, country={country!r} "
                f"in {path}. Available topics: {available_topics}"
            )
        if len(rows) > 1:
            raise ValueError(
                f"Found {len(rows)} rows for topic={topic!r}, country={country!r} in {path}."
            )

        features = parse_feature_ids(rows.iloc[0]["salient_features"])
        if num_features is not None:
            features = features[:num_features]
        if not features:
            if skip_missing:
                print(
                    f"[skip] Empty salient-feature list for topic={topic!r}, "
                    f"country={country!r} in {path}."
                )
                continue
            raise ValueError(
                f"Empty salient-feature list for topic={topic!r}, country={country!r} in {path}."
            )
        output[topic] = tuple(features)

    return output
