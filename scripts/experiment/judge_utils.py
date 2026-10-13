"""Shared prompt and parsing helpers for LLM-as-judge scripts."""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple


SYSTEM_MESSAGE = "You are an intelligent, helpful, and logical assistant."


def judge_messages(judge_prompt: str, answer: str, culture: str) -> List[Dict[str, str]]:
    question = f"""input:
culture_to_check: "{culture}"
response: "{answer}"
output:
"""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": judge_prompt + question},
    ]


def chat_prompt(tokenizer, judge_prompt: str, answer: str, culture: str) -> str:
    return tokenizer.apply_chat_template(
        judge_messages(judge_prompt, answer, culture),
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_judge_json_relaxed(text: str) -> Dict[str, str]:
    try:
        parsed = json.loads(text)
    except Exception:
        return {"Decision": "None", "Reasoning": text if text else "None"}

    if not isinstance(parsed, dict):
        return {"Decision": "None", "Reasoning": text if text else "None"}
    return parsed


def normalize_decision(value: object) -> str:
    decision = str(value or "").strip().lower()
    if "yes" in decision:
        return "Yes"
    if "no" in decision:
        return "No"
    return "No"


def records_from_completion_choices(choices, prompts_count: int) -> Tuple[int, List[List[str]]]:
    ordered = sorted(choices, key=lambda choice: choice.index)
    if len(ordered) != prompts_count:
        raise RuntimeError(f"Expected {prompts_count} completions, got {len(ordered)}.")

    biased_count = 0
    records: List[List[str]] = []
    for choice in ordered:
        text = (choice.text or "").strip()
        if not text:
            records.append(["No", "No Clear Text Generated"])
            continue

        parsed = parse_judge_json_relaxed(text)
        decision = normalize_decision(parsed.get("Decision"))
        reasoning = str(parsed.get("Reasoning", text))
        if decision == "Yes":
            biased_count += 1
        records.append([decision, reasoning])

    return biased_count, records


def judge_answers_for_country(
    client,
    tokenizer,
    model_id: str,
    judge_prompt: str,
    answers: Sequence[str],
    check_country: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[int, List[List[str]], Sequence[str]]:
    prompts = [
        chat_prompt(tokenizer, judge_prompt, answer, check_country)
        for answer in answers
    ]
    print(
        f"[info] Inference started for judge country={check_country} "
        f"| prompts={len(prompts)}"
    )

    response = client.completions.create(
        model=model_id,
        prompt=prompts,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    biased_count, records = records_from_completion_choices(response.choices, len(prompts))
    return biased_count, records, answers
