"""Prompt exported from Meta's Llama 3.1 8B Instruct GSM8K evaluations."""

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path


PROMPT_PATH = Path(__file__).with_name("prompts") / "meta_llama_3_1_8b_instruct_gsm8k_8shot.json"


def load_meta_prompt():
    source = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    if source["shots"] != 8 or source["template"].count("{{question}}") != 1:
        raise ValueError(f"Invalid Meta GSM8K template: {PROMPT_PATH}")
    source["template_sha256"] = hashlib.sha256(source["template"].encode()).hexdigest()
    return source


def build_meta_prompt(question, source):
    return source["template"].replace("{{question}}", question)


def tokenize_meta_prompt(tokenizer, prompt, max_input_length):
    # Published prompts already include all chat headers but omit the BOS token.
    # Never apply a chat template again or silently truncate demonstrations.
    if not tokenizer.bos_token or tokenizer.bos_token_id is None:
        raise ValueError("Meta GSM8K prompts require a Llama tokenizer with a BOS token")
    enc = tokenizer(tokenizer.bos_token + prompt, add_special_tokens=False, return_tensors="pt")
    if enc["input_ids"].shape[-1] > max_input_length:
        raise ValueError(
            f"Meta GSM8K prompt has {enc['input_ids'].shape[-1]} tokens, exceeding "
            f"--max_input_length {max_input_length}; increase the limit to preserve the full prompt"
        )
    return enc


def extract_meta_answer(text):
    """Parse a numeric answer following Meta's final-answer marker."""
    matches = list(re.finditer(r"The final answer is\s+", text))
    if not matches:
        return None
    final = text[matches[-1].end():].strip()
    match = re.fullmatch(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)[.]?", final)
    return Decimal(match.group(1).replace(",", "")) if match else None
