"""Generate named and countryless responses to the cultural questions.

Writes generation CSVs under DATA_DIR/generations/{with_name,woname}."""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from dotenv import dotenv_values
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generation_utils import (
    build_system_prompt,
    chat_text,
    get_questions,
    load_fb_names_json,
    resolve_country_name,
)

ENV = dotenv_values(Path(ROOT) / ".env")
DATA_DIR = Path(ENV["DATA_DIR"])


MODEL_ID = ENV["GENERATION_MODEL_ID"]

OUT_DIR = DATA_DIR / "generations"
FB_NAMES_JSON = Path("external/data/fb_names.json")

MODES = ["with_name", "woname"]
PERSONS = ["First Person"]
FACETS = ["food", "clothing", "tradition", "rituals"]

MAX_NAMES_PER_COUNTRY = 30
BATCH_SIZE = 512
MAX_NEW_TOKENS = 2048
TEMPERATURE = float(ENV["GENERATION_TEMPERATURE"])
TOP_P = float(ENV["GENERATION_TOP_P"])
TARGET_COUNTRIES_RAW = ENV["TARGET_COUNTRIES"]
MAX_MODEL_LEN = 8096

# vLLM-specific
VLLM_DTYPE = "float16"   # "bfloat16", "float16", or "auto"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.9
TRUST_REMOTE_CODE = True


def batched(iterable: Iterable, batch_size: int) -> Iterator[List]:
    """
    Groups items from an iterable into batches of size batch_size.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_target_countries(raw_value: str) -> List[str]:
    """
    Parses TARGET_COUNTRIES from .env.
    Accepts semicolon-separated country names.
    """
    return [
        country.strip()
        for country in str(raw_value or "").split(";")
        if country.strip()
    ]


def filter_names_by_country(
    names_by_country: Dict[str, List[str]],
    target_countries: List[str],
) -> Dict[str, List[str]]:
    """
    Keeps only the requested countries, resolving case-insensitive country names.
    """
    if not target_countries:
        return names_by_country

    resolved_countries: List[str] = []
    for requested_country in target_countries:
        country = resolve_country_name(requested_country, names_by_country)
        if country not in resolved_countries:
            resolved_countries.append(country)

    return {
        country: names_by_country[country]
        for country in resolved_countries
    }


def iter_tasks(
    mode: str,
    person: str,
    facet: str,
    names_by_country: Dict[str, List[str]],
) -> Iterator[Tuple[str, Optional[str], str, str]]:
    """
    Yields tasks of the form:
        (country, name, question, sys_prompt)
    """
    questions = get_questions(facet)

    if mode == "woname":
        sys_p = build_system_prompt(mode, person, None, None)
        for q in questions:
            yield ("", None, q, sys_p)

    elif mode == "with_country":
        for country in names_by_country.keys():
            sys_p = build_system_prompt(mode, person, None, country)
            for q in questions:
                yield (country, None, q, sys_p)

    elif mode == "with_name":
        for country, names in names_by_country.items():
            for name in names:
                sys_p = build_system_prompt(mode, person, name, None)
                for q in questions:
                    yield (country, name, q, sys_p)

    elif mode == "name_and_country":
        for country, names in names_by_country.items():
            for name in names:
                sys_p = build_system_prompt(mode, person, name, country)
                for q in questions:
                    yield (country, name, q, sys_p)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def count_tasks(
    mode: str,
    facet: str,
    names_by_country: Dict[str, List[str]],
) -> int:
    """
    Returns the total number of tasks for a given mode/facet.
    Useful for tqdm total.
    """
    n_questions = len(get_questions(facet))
    n_countries = len(names_by_country)
    n_names = sum(len(v) for v in names_by_country.values())

    if mode == "woname":
        return n_questions
    if mode == "with_country":
        return n_countries * n_questions
    if mode == "with_name":
        return n_names * n_questions
    if mode == "name_and_country":
        return n_names * n_questions

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facets", nargs="+", choices=FACETS, default=FACETS)
    args = parser.parse_args()
    mdl_short = ENV["GENERATION_MODEL_SHORT"]

    print("[info] Loading country/name data...")
    names_by_country = load_fb_names_json(FB_NAMES_JSON, MAX_NAMES_PER_COUNTRY)
    if not names_by_country:
        raise FileNotFoundError(
            f"No countries/names loaded from {FB_NAMES_JSON}. "
            "Check path/permissions or adjust load_fb_names_json()."
        )

    target_countries = parse_target_countries(TARGET_COUNTRIES_RAW)
    total_countries = len(names_by_country)
    total_names = sum(len(names) for names in names_by_country.values())
    names_by_country = filter_names_by_country(names_by_country, target_countries)
    selected_countries = len(names_by_country)
    selected_names = sum(len(names) for names in names_by_country.values())
    if target_countries:
        print(
            "[info] Filtering generation to target countries: "
            + ", ".join(names_by_country.keys())
        )
        print(
            "[info] Selected "
            f"{selected_countries}/{total_countries} countries and "
            f"{selected_names}/{total_names} names; leaving outside "
            f"{total_countries - selected_countries} countries and "
            f"{total_names - selected_names} names."
        )
    else:
        print(
            f"[info] TARGET_COUNTRIES is not set; using all "
            f"{total_countries} countries and {total_names} names."
        )


    print(f"[info] Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_REMOTE_CODE)
    tok.padding_side = "left"

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"[info] Loading vLLM model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        dtype=VLLM_DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
    )

    for facet in args.facets:
        for person in PERSONS:
            for mode in MODES:
                out_folder = OUT_DIR / mode
                out_folder.mkdir(parents=True, exist_ok=True)

                csv_path = out_folder / f"{mdl_short}_{facet}_{person.replace(' ', '')}_{mode}.csv"

                is_new = not csv_path.exists()
                if is_new:
                    with csv_path.open("w", encoding="utf-8", newline="") as fout:
                        w = csv.writer(fout)
                        w.writerow(["country", "sys_prompt", "name", "question", "output"])

                total_tasks = count_tasks(mode, facet, names_by_country)
                task_iter = iter_tasks(mode, person, facet, names_by_country)

                desc = f"{mdl_short}|{facet}|{person}|{mode}"
                with csv_path.open("a", encoding="utf-8", newline="") as fout:
                    w = csv.writer(fout)

                    pbar = tqdm(total=total_tasks, desc=desc)
                    for batch in batched(task_iter, BATCH_SIZE):
                        prompts: List[str] = []
                        metas: List[Tuple[str, Optional[str], str, str]] = []

                        for country, name, question, sys_p in batch:
                            prompts.append(chat_text(tok, sys_p, question))
                            metas.append((country, name, question, sys_p))

                        outputs = llm.generate(prompts, sampling_params)

                        for output_obj, (country, name, question, sys_p) in zip(outputs, metas):
                            if not output_obj.outputs:
                                text_out = ""
                            else:
                                text_out = output_obj.outputs[0].text


                            w.writerow([
                                country or "",
                                sys_p,
                                name or "",
                                question,
                                text_out,
                            ])

                        fout.flush()
                        pbar.update(len(batch))

                    pbar.close()

    print("Generation completed.")


if __name__ == "__main__":
    main()
