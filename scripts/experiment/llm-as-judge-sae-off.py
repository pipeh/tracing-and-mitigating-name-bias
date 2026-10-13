"""Judge each SAE-off response against the intervention's country.

Both named and countryless files are country-specific. Writes TSVs under
DATA_DIR/analysis-sae-off/<judge>/."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from transformers import AutoTokenizer


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from external import prompt_updated as P
from judge_utils import judge_answers_for_country as run_judge_answers_for_country

ENV = dotenv_values(Path(ROOT) / ".env")
DATA_DIR = Path(ENV["DATA_DIR"])


JUDGE_MODEL = ENV["JUDGE_MODEL_ID"]
VLLM_BASE_URL = ENV["JUDGE_VLLM_BASE_URL"]
JUDGE_API_KEY = ENV["JUDGE_API_KEY"]
TOKENIZER_ID = ENV["JUDGE_TOKENIZER_ID"]
GENERATIONS_DIR = DATA_DIR / "generations-sae-off"
ANALYSIS_DIR = DATA_DIR / "analysis-sae-off"
TARGET_MODEL_SHORT = ENV["GENERATION_MODEL_SHORT"]
DEFAULT_MODES = ["with_name", "woname"]
SUPPORTED_MODES = ["with_name", "with_country", "name_and_country", "woname"]

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_CHUNK_SIZE = 192

PAIR_COLUMNS = [
    "country",
    "check_country",
    "answer_index",
    "name",
    "question",
    "answer",
    "decision",
    "reasoning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate country-specific SAE-off generation CSVs using a vLLM "
            "OpenAI-compatible judge server."
        )
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_MODES,
        default=DEFAULT_MODES,
        help="Generation mode directories to evaluate.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate pairs whose TSV output already exists.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    judge_path = Path(JUDGE_MODEL)
    if not JUDGE_MODEL.strip() or judge_path.name != JUDGE_MODEL:
        parser.error(
            "JUDGE_MODEL_ID must be a single directory/model name without path separators."
        )
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be greater than zero.")
    if args.chunk_size <= 0:
        parser.error("--chunk_size must be greater than zero.")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative.")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1].")

    args.modes = unique_preserving_order(args.modes)
    return args


def unique_preserving_order(values: Iterable[str]) -> List[str]:
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
    return cleaned.strip("_") or "unknown"


def chunked_iterable(iterable: Sequence[str], chunk_size: int):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, chunk_size)):
        yield chunk


def sidecar_path(csv_path: Path) -> Path:
    return Path(str(csv_path) + ".meta.json")


def load_generation_metadata(csv_path: Path) -> Dict:
    path = sidecar_path(csv_path)
    if not path.exists():
        return {}

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Could not read metadata sidecar {path}: {exc}")
        return {}

    if not isinstance(metadata, dict):
        print(f"[warn] Metadata sidecar is not a JSON object: {path}")
        return {}
    return metadata


def target_model_short_from_metadata(
    metadata: Dict,
    fallback: Optional[str] = None,
) -> str:
    model_short = str(metadata.get("model_short", "") or fallback or "").strip()
    if model_short:
        return model_short
    model_id = str(metadata.get("model_id", "")).strip()
    if model_id:
        return model_id.split("/")[-1]

    raise ValueError("Generation metadata is missing required field: model_id")


def discover_generation_files(
    generations_dir: Path,
    judge_model: str,
    modes: Sequence[str],
    target_model_short: str,
) -> List[Path]:
    judge_dir = generations_dir / judge_model
    files: List[Path] = []

    if not judge_dir.exists():
        raise FileNotFoundError(
            f"Judge generation directory not found: {judge_dir}"
        )

    for mode in modes:
        mode_dir = judge_dir / mode
        if not mode_dir.exists():
            print(f"[warn] Mode directory not found, skipping: {mode_dir}")
            continue

        pattern = f"{target_model_short}_*.csv" if target_model_short else "*.csv"
        mode_files = sorted(path for path in mode_dir.glob(pattern) if path.is_file())
        print(f"[info] Discovered {len(mode_files)} CSVs in {mode_dir}")
        files.extend(mode_files)

    return files


def generation_countries(data: pd.DataFrame, metadata: Dict) -> pd.Series:
    if "country" not in data.columns:
        data["country"] = ""

    countries = data["country"].fillna("").astype(str).str.strip()
    metadata_country = str(metadata.get("country", "")).strip()

    if metadata_country:
        countries = countries.mask(countries == "", metadata_country)

    return countries


# Judge prompt and parsing


def judge_answers_for_country(
    client: OpenAI,
    tokenizer,
    model_id: str,
    answers: Sequence[str],
    check_country: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[int, List[List[str]], List[str]]:
    biased_count, records, _ = run_judge_answers_for_country(
        client=client,
        tokenizer=tokenizer,
        model_id=model_id,
        judge_prompt=P.BIAS_CHECK_SYS_PROMPT,
        answers=answers,
        check_country=check_country,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return biased_count, records, list(answers)


# Evaluation

def evaluate_generation_file(
    csv_path: Path,
    args: argparse.Namespace,
    client: OpenAI,
    tokenizer,
) -> None:
    metadata = load_generation_metadata(csv_path)
    mode = str(metadata.get("mode", csv_path.parent.name)).strip()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Could not resolve a supported mode for {csv_path}; got {mode!r}."
        )

    try:
        data = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Error loading generation file {csv_path}: {exc}") from exc

    required_columns = {"output", "question"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}"
        )

    data = data.copy()
    data["_source_answer_index"] = data.index
    data["output"] = data["output"].fillna("").astype(str)
    data["question"] = data["question"].fillna("").astype(str)

    if "name" not in data.columns:
        data["name"] = ""
    data["name"] = data["name"].fillna("").astype(str)
    data["country"] = generation_countries(data, metadata)

    if (data["country"] == "").any():
        missing_rows = int((data["country"] == "").sum())
        raise ValueError(
            f"Could not resolve the generation country for {missing_rows} rows in "
            f"{csv_path}. Expected the CSV country column or sidecar metadata country."
        )

    target_model_short = target_model_short_from_metadata(
        metadata,
        fallback=TARGET_MODEL_SHORT,
    )
    output_folder = (
        ANALYSIS_DIR / JUDGE_MODEL / mode / target_model_short
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    grouped_data = data.groupby("country", sort=False, dropna=False)
    for generation_country, group in grouped_data:
        generation_country = str(generation_country).strip()
        check_country = generation_country

        row_dicts = group[
            ["output", "name", "question", "_source_answer_index"]
        ].to_dict("records")
        outputs = [row["output"] for row in row_dicts]

        pair_output_filename = output_folder / (
            f"{csv_path.stem}_"
            f"{slugify(generation_country)}_"
            f"{slugify(check_country)}.tsv"
        )

        if pair_output_filename.exists() and not args.overwrite:
            print(f"[skip] Existing pair file: {pair_output_filename}")
            continue

        print(
            f"[info] Evaluating file={csv_path.name} | mode={mode} | "
            f"generation_country={generation_country} | rows={len(outputs)}"
        )

        bias_details = []
        total_biased_count = 0
        total_count = 0

        for chunk in chunked_iterable(outputs, args.chunk_size):
            biased_count, records, original_answers = judge_answers_for_country(
                client=client,
                tokenizer=tokenizer,
                model_id=JUDGE_MODEL,
                answers=chunk,
                check_country=check_country,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            chunk_start = total_count
            total_biased_count += biased_count
            total_count += len(chunk)

            for index, answer in enumerate(original_answers):
                row_info = row_dicts[chunk_start + index]
                decision, reasoning = records[index]
                bias_details.append(
                    {
                        "country": generation_country,
                        "check_country": check_country,
                        "answer_index": row_info["_source_answer_index"],
                        "name": row_info["name"],
                        "question": row_info["question"],
                        "answer": answer,
                        "decision": decision,
                        "reasoning": reasoning,
                    }
                )

        pair_df = pd.DataFrame(bias_details, columns=PAIR_COLUMNS)
        pair_df.to_csv(
            pair_output_filename,
            index=False,
            sep="\t",
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            encoding="utf-8",
        )

        print(
            f"[saved] {pair_output_filename} | "
            f"biased={total_biased_count}/{total_count}"
        )


def main() -> None:
    args = parse_args()

    generation_files = discover_generation_files(
        generations_dir=GENERATIONS_DIR,
        judge_model=JUDGE_MODEL,
        modes=args.modes,
        target_model_short=TARGET_MODEL_SHORT,
    )
    if not generation_files:
        raise RuntimeError(
            "No SAE-off generation CSVs were discovered. Check DATA_DIR, "
            "JUDGE_MODEL_ID, --modes, and GENERATION_MODEL_SHORT."
        )

    print(f"[info] Generation files to evaluate: {len(generation_files)}")
    for path in generation_files:
        print(f"  - {path}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] Connecting to vLLM server: {VLLM_BASE_URL}")
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=JUDGE_API_KEY,
        timeout=3600.0,
        max_retries=3,
    )

    print(f"[info] Using served judge model: {JUDGE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        trust_remote_code=args.trust_remote_code,
        local_files_only=False,
    )

    for csv_path in generation_files:
        try:
            evaluate_generation_file(
                csv_path=csv_path,
                args=args,
                client=client,
                tokenizer=tokenizer,
            )
        except Exception as exc:
            print(f"[error] {csv_path}: {exc}")

    print("Evaluation completed.")


if __name__ == "__main__":
    main()
