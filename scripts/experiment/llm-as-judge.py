"""Judge named responses against their country and shared baselines against
each target country. Writes TSVs under DATA_DIR/analysis/<judge>/."""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
from pathlib import Path
from itertools import islice
from typing import List, Tuple

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI
from transformers import AutoTokenizer

from external import prompt_updated as P
from judge_utils import judge_answers_for_country as run_judge_answers_for_country

ENV = dotenv_values(Path(ROOT) / ".env")
DATA_DIR = Path(ENV["DATA_DIR"])


GENERATIONS_BASE = DATA_DIR / "generations"

ROOT_ANALYSIS = DATA_DIR / "analysis"


MODEL_ID = ENV["JUDGE_MODEL_ID"]

VLLM_BASE_URL = ENV["JUDGE_VLLM_BASE_URL"]
JUDGE_API_KEY = ENV["JUDGE_API_KEY"]

TOKENIZER_ID = ENV["JUDGE_TOKENIZER_ID"]

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0
TOP_P = 1.0

TRUST_REMOTE_CODE = True

CHUNK_SIZE = 384

MODES = ["woname", "with_name"]
FACETS = ["food", "clothing", "tradition", "rituals"]  # e.g. ["food", "clothing", "rituals"]
PERSONS = ["FirstPerson"]         # filename version, not "First Person"
TARGET_MODEL_SHORT = ENV["GENERATION_MODEL_SHORT"]

# Countries used as judge targets
countries = [
    "Canada",
    "Switzerland",
    "Ireland",
    "Indonesia",
    "United States",
    "Germany",
    "Iran, Islamic Republic of",
    "South Africa",
    "Peru",
    "Korea, Republic of",
    "Japan",
    "Philippines",
    "Russian Federation",
    "Finland",
    "Egypt",
    "Brazil",
    "Spain",
    "United Kingdom",
    "Portugal",
    "Sweden",
    "Poland",
    "Greece",
    "Morocco",
    "China",
    "Mexico",
    "Israel",
    "Italy",
    "Türkiye",
    "India",
    "France",
]

if ENV.get("TARGET_COUNTRIES"):
    countries = [country.strip() for country in ENV["TARGET_COUNTRIES"].split(";") if country.strip()]

JUDGE_NAME = MODEL_ID.replace("/", "_")

ROOT_ANALYSIS.mkdir(parents=True, exist_ok=True)

ANALYSIS_FOLDER_WITH_NAME = ROOT_ANALYSIS / JUDGE_NAME / 'with_name'
ANALYSIS_FOLDER_WITHOUT_NAME = ROOT_ANALYSIS / JUDGE_NAME / 'woname'
ANALYSIS_FOLDER_WITH_NAME.mkdir(parents=True, exist_ok=True)
ANALYSIS_FOLDER_WITHOUT_NAME.mkdir(parents=True, exist_ok=True)

print(f"[info] Connecting to vLLM server: {VLLM_BASE_URL}")

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=JUDGE_API_KEY,
    timeout=3600.0,
    max_retries=3,
)

print(f"[info] Using served judge model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID,
    trust_remote_code=TRUST_REMOTE_CODE,
    local_files_only=False,
)


def build_input_files(
    base_dir: Path,
    model_short: str,
    facets: List[str],
    persons: List[str],
    modes: List[str],
) -> List[Path]:
    """List generation CSVs for the requested modes, facets, and personas."""
    files: List[Path] = []

    for mode in modes:
        folder = base_dir / mode
        for facet in facets:
            for person in persons:
                file_path = folder / f"{model_short}_{facet}_{person}_{mode}.csv"
                files.append(file_path)

    return files


def judge_answers_for_country(
    answers: List[str],
    check_country: str,
) -> Tuple[int, List[List[str]], List[str]]:
    return run_judge_answers_for_country(
        client=client,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        judge_prompt=P.BIAS_CHECK_SYS_PROMPT,
        answers=answers,
        check_country=check_country,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )


def chunked_iterable(iterable, chunk_size):
    """Yield successive chunk_size chunks from iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, chunk_size)):
        yield chunk


files = build_input_files(
    base_dir=GENERATIONS_BASE,
    model_short=TARGET_MODEL_SHORT,
    facets=FACETS,
    persons=PERSONS,
    modes=MODES,
)

print("[info] Files to evaluate:")
for f in files:
    print(f"  - {f}")


for file in files:
    if not file.exists():
        print(f"[warn] File not found, skipping: {file}")
        continue

    try:
        data = pd.read_csv(file)
        # Keep the row number from the original generations CSV for traceability.
        data["_source_answer_index"] = data.index
    except Exception as e:
        print(f"[error] Error loading file {file}: {e}")
        continue

    required_cols = {"output", "question"}
    missing = required_cols - set(data.columns)
    if missing:
        print(f"[error] Missing required columns in {file}: {missing}")
        continue

    if "name" not in data.columns:
        data["name"] = ""
    if "country" not in data.columns:
        data["country"] = ""

    file_name = "_".join(file.name.split("_")[1:3])

    outputs_by_country = {}

    # Baseline woname usually has empty country values
    country_series = data["country"].fillna("").astype(str)

    if (country_series == "").all():
        outputs_by_country["baseline"] = data[
            ["output", "name", "question", "_source_answer_index"]
        ].to_dict("records")
    else:
        grouped_data = data.groupby("country", dropna=False)
        for country, group in grouped_data:
            country_key = "" if pd.isna(country) else str(country)
            if country_key == "":
                country_key = "baseline"
            elif country_key not in countries:
                print(f"[warn] Country '{country_key}' not in predefined list, skipping.")
                continue
            outputs_by_country[country_key] = group[
                ["output", "name", "question", "_source_answer_index"]
            ].to_dict("records")

    for generation_country, row_dicts in outputs_by_country.items():
        outputs = [rd["output"] for rd in row_dicts]

        # with_name: evaluate each country's generations against itself.
        # woname baseline: evaluate the same baseline responses against every country.
        if generation_country == "baseline":
            check_countries = countries
        else:
            check_countries = [generation_country]

        print(
            f"[info] Processing {len(outputs)} outputs "
            f"for generation_country={generation_country}"
        )

        for check_country in check_countries:
            print(
                f"[info] Evaluating generation_country={generation_country} "
                f"against check_country={check_country}"
            )

            if "with_name" in str(file):
                output_folder = ANALYSIS_FOLDER_WITH_NAME / TARGET_MODEL_SHORT
            else:
                output_folder = ANALYSIS_FOLDER_WITHOUT_NAME / TARGET_MODEL_SHORT

            output_folder.mkdir(parents=True, exist_ok=True)

            pair_output_filename = (
                output_folder
                / (
                    f"{file_name}_"
                    f"{generation_country}_"
                    f"{check_country}.tsv"
                )
            )

            if pair_output_filename.exists():
                print(f"[info] Skipping existing pair file: {pair_output_filename}")
                continue

            bias_details = []

            total_count = 0

            for chunk in chunked_iterable(outputs, CHUNK_SIZE):
                _, recs, original_answers = (
                    judge_answers_for_country(
                        chunk,
                        check_country,
                    )
                )

                chunk_start = total_count
                total_count += len(chunk)

                for i, answer in enumerate(original_answers):
                    current_idx = chunk_start + i
                    row_info = row_dicts[current_idx]
                    element = recs[i]

                    bias_details.append({
                        "country": generation_country,
                        "check_country": check_country,
                        "answer_index": row_info["_source_answer_index"],
                        "name": row_info["name"],
                        "question": row_info["question"],
                        "answer": answer,
                        "decision": element[0],
                        "reasoning": element[1],
                    })

            pair_df = pd.DataFrame(bias_details)

            pair_df.to_csv(
                pair_output_filename,
                index=False,
                sep="\t",
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
                encoding="utf-8",
            )

            print(f"[info] Saved partial results to {pair_output_filename}")
