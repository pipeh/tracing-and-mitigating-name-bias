"""Generate responses with CRISP-selected SAE features disabled across layers.

The first N features from each available layer form a simultaneous
intervention. Missing layers are skipped; topics with no layers are omitted."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import dotenv_values
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SAE_ANALYSIS_DIR = os.path.join(ROOT, "scripts", "sae-analysis")
if SAE_ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, SAE_ANALYSIS_DIR)

from generation_utils import LayerSpec, build_tasks, chat_text, load_fb_names_json, load_layer_feature_rows, parse_layer_spec, resolve_country_name, resolve_device, slugify, torch_dtype_from_name, unique_preserving_order
from utils import JumpReLU, load_llamascope_sae

ENV = dotenv_values(Path(ROOT) / ".env")
DATA_DIR = Path(ENV["DATA_DIR"])
CRISP_DATA_DIR = DATA_DIR / "crisp"


DEFAULT_MODEL_ID = ENV["GENERATION_MODEL_ID"]
MODEL = ENV["GENERATION_MODEL_SHORT"]
JUDGE_MODEL = ENV["JUDGE_MODEL_ID"]
DEFAULT_LAYERS = [
    layer
    for layer in str(ENV["LAYERS"]).replace(",", " ").split()
    if layer
]
DEFAULT_FB_NAMES_JSON = Path("external/data/fb_names.json")
DEFAULT_FEATURES_DIR = CRISP_DATA_DIR / JUDGE_MODEL
DEFAULT_OUT_DIR = DATA_DIR / "generations-sae-off" / JUDGE_MODEL
DEFAULT_NUM_FEATURES = int(ENV["TOP_N"])

DEFAULT_MODES = ["with_name", "woname"]
SUPPORTED_MODES = ["with_name", "with_country", "name_and_country", "woname"]
TOPICS = ["food", "tradition", "clothing", "rituals"]
DEFAULT_PERSON = "First Person"
DEFAULT_SITE = ENV["SITE"]
DEFAULT_EXPANSION = ENV["EXPANSION"]

DEFAULT_MAX_NAMES_PER_COUNTRY = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_INPUT_LENGTH = 6048
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = float(ENV["GENERATION_TEMPERATURE"])
DEFAULT_TOP_P = float(ENV["GENERATION_TOP_P"])
DEFAULT_SEED = 0
DEFAULT_DTYPE = "float16"

CSV_COLUMNS = [
    "country",
    "sys_prompt",
    "name",
    "question",
    "output",
]


# Data structures

@dataclass(frozen=True)
class ExperimentPlan:
    layer_spec: LayerSpec
    topic: str
    mode: str
    feature_ids: Tuple[int, ...]
    features_file: Path
    output_file: Path


@dataclass(frozen=True)
class MultiLayerExperimentPlan:
    topic: str
    mode: str
    layer_features: Tuple[Tuple[LayerSpec, Tuple[int, ...]], ...]
    feature_files: Tuple[Tuple[LayerSpec, Path], ...]
    output_file: Path

    @property
    def n_features(self) -> int:
        return sum(len(feature_ids) for _layer, feature_ids in self.layer_features)


# Argument parsing

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate responses for multiple topics and LlamaScope layers while "
            "disabling the salient features of one country."
        )
    )

    parser.add_argument(
        "--country",
        required=True,
        help="Country whose topic-country salient features will be disabled.",
    )
    parser.add_argument("--topics", nargs="+", choices=TOPICS, help="Facets to generate (default: all four).")
    parser.add_argument(
        "--layers",
        nargs="+",
        default=DEFAULT_LAYERS,
        help=(
            "Layers whose per-layer salient-feature files will be loaded. "
            "Defaults to LAYERS from .env."
        ),
    )
    parser.add_argument(
        "--site",
        default=DEFAULT_SITE,
        choices=["R"],
        help=(
            "Default SAE site for layer values without a suffix. Only residual "
            "stream site R is supported because the intervention hooks layer outputs."
        ),
    )
    parser.add_argument(
        "--expansion",
        default=DEFAULT_EXPANSION,
        choices=["8x", "32x"],
        help="LlamaScope SAE expansion and parquet filename suffix.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=SUPPORTED_MODES,
        help="Prompt conditions to generate for every layer-topic combination.",
    )
    parser.add_argument(
        "--person",
        default=DEFAULT_PERSON,
        choices=[DEFAULT_PERSON],
        help="Prompt persona. The current Pawar pipeline uses First Person.",
    )

    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fb_names_json", type=Path, default=DEFAULT_FB_NAMES_JSON)

    parser.add_argument(
        "--num_features",
        type=int,
        default=DEFAULT_NUM_FEATURES,
        help=(
            "Disable the first N salient features from each layer's parquet. "
            "Defaults to TOP_N from .env."
        ),
    )
    parser.add_argument(
        "--max_names_per_country",
        type=int,
        default=DEFAULT_MAX_NAMES_PER_COUNTRY,
        help="Maximum number of male and female names loaded per country.",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=DEFAULT_MAX_INPUT_LENGTH,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSVs that already exist. Otherwise they are skipped.",
    )
    parser.add_argument(
        "--trust_remote_code",
        dest="trust_remote_code",
        action="store_true",
        help="Allow trust_remote_code when loading the model and tokenizer.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code.",
    )
    parser.set_defaults(topics=TOPICS, trust_remote_code=True)

    args = parser.parse_args()

    if args.num_features is not None and args.num_features <= 0:
        parser.error("--num_features must be greater than zero.")
    if args.max_names_per_country <= 0:
        parser.error("--max_names_per_country must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than zero.")
    if args.max_input_length <= 0:
        parser.error("--max_input_length must be greater than zero.")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens must be greater than zero.")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative.")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in (0, 1].")

    args.topics = unique_preserving_order(topic.lower() for topic in args.topics)
    args.modes = unique_preserving_order(args.modes)
    if args.layers:
        args.layers = unique_preserving_order(args.layers)
    return args


def salient_features_path(
    features_dir: Path,
    layer_spec: LayerSpec,
    expansion: str,
) -> Path:
    return features_dir / (
        f"topic_country_salient_features_{MODEL}_"
        f"{layer_spec.layer_id}_{expansion}.parquet"
    )


def parse_string_list(value, column_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse {column_name}: {value!r}") from exc
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected {column_name} to be a list, got {type(value).__name__}.")
    return [str(item) for item in value]


@dataclass
class SelectedSAEIntervention:
    layer_spec: LayerSpec
    feature_ids: Tuple[int, ...]
    encoder_weight: torch.Tensor
    encoder_bias: Optional[torch.Tensor]
    decoder_weight: torch.Tensor
    thresholds: Optional[torch.Tensor]
    activation_kind: str


def extract_selected_intervention(
    sae,
    layer_spec: LayerSpec,
    feature_ids: Sequence[int],
) -> SelectedSAEIntervention:
    """Keep only selected SAE rows/columns so full SAEs need not remain in memory."""
    features = tuple(unique_preserving_order(int(feature) for feature in feature_ids))
    if not features:
        raise ValueError(f"No feature ids supplied for {layer_spec.layer_id}.")
    if not isinstance(getattr(sae, "enc", None), torch.nn.Linear):
        raise TypeError("Selective intervention requires sae.enc to be nn.Linear.")
    if not isinstance(getattr(sae, "dec", None), torch.nn.Linear):
        raise TypeError("Selective intervention requires sae.dec to be nn.Linear.")

    parameter = next(sae.parameters())
    device, dtype = parameter.device, parameter.dtype
    d_sae = int(sae.enc.weight.shape[0])
    invalid = [feature for feature in features if feature < 0 or feature >= d_sae]
    if invalid:
        raise ValueError(
            f"Feature ids outside {layer_spec.layer_id} SAE range [0, {d_sae - 1}]: {invalid}"
        )

    ids = torch.tensor(features, device=device, dtype=torch.long)
    encoder_weight = sae.enc.weight.index_select(0, ids).detach().clone()
    encoder_bias = None
    if sae.enc.bias is not None:
        encoder_bias = sae.enc.bias.index_select(0, ids).detach().clone()
    decoder_weight = sae.dec.weight.index_select(1, ids).detach().clone()

    activation = sae.act
    thresholds = None
    activation_kind = "relu"
    if hasattr(activation, "get_jumprelu_threshold"):
        thresholds = (
            activation.get_jumprelu_threshold()
            .reshape(-1)
            .index_select(0, ids)
            .to(device=device, dtype=dtype)
            .detach()
            .clone()
        )
        activation_kind = "jumprelu"
    elif hasattr(activation, "log_jumprelu_threshold"):
        thresholds = (
            activation.log_jumprelu_threshold.exp()
            .reshape(-1)
            .index_select(0, ids)
            .to(device=device, dtype=dtype)
            .detach()
            .clone()
        )
        activation_kind = "jumprelu"
    elif isinstance(activation, JumpReLU):
        thresholds = torch.tensor(activation.threshold, device=device, dtype=dtype)
        activation_kind = "shifted_relu"
    elif "relu" not in activation.__class__.__name__.lower():
        raise TypeError(
            f"Unsupported SAE activation for lightweight multilayer intervention: "
            f"{activation.__class__.__name__}."
        )

    return SelectedSAEIntervention(
        layer_spec=layer_spec,
        feature_ids=features,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        decoder_weight=decoder_weight,
        thresholds=thresholds,
        activation_kind=activation_kind,
    )


def build_selected_intervention_hook(intervention: SelectedSAEIntervention):
    def hook_fn(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        if hidden_states is None:
            return output

        pre = F.linear(
            hidden_states,
            intervention.encoder_weight,
            intervention.encoder_bias,
        )
        if intervention.activation_kind == "jumprelu":
            selected = pre * (pre > intervention.thresholds)
        elif intervention.activation_kind == "shifted_relu":
            selected = F.relu(pre - intervention.thresholds)
        else:
            selected = F.relu(pre)
        contribution = selected @ intervention.decoder_weight.T
        modified = hidden_states - contribution

        if isinstance(output, tuple):
            return (modified, *output[1:])
        if isinstance(output, list):
            return [modified, *output[1:]]
        return modified

    return hook_fn


@contextmanager
def multilayer_disabled_context(
    model,
    interventions: Sequence[SelectedSAEIntervention],
):
    handles = []
    try:
        for intervention in interventions:
            handle = transformer_layer(
                model, intervention.layer_spec.layer
            ).register_forward_hook(build_selected_intervention_hook(intervention))
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def generate_responses_multilayer(
    model,
    tokenizer,
    interventions: Sequence[SelectedSAEIntervention],
    tasks: Sequence[Tuple[str, Optional[str], str, str]],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    description: str,
) -> List[str]:
    device = next(model.parameters()).device
    set_seed(seed)
    outputs: List[str] = []
    do_sample = temperature > 0
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    else:
        generation_kwargs.update(temperature=None, top_p=None)

    with multilayer_disabled_context(model, interventions):
        for start in tqdm(
            range(0, len(tasks), batch_size),
            desc=description,
            unit="batch",
        ):
            batch = tasks[start : start + batch_size]
            prompts = [
                chat_text(tokenizer, system_prompt, question)
                for _, _, question, system_prompt in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs)
            outputs.extend(
                text.strip()
                for text in tokenizer.batch_decode(
                    generated[:, input_width:], skip_special_tokens=True
                )
            )

    if len(outputs) != len(tasks):
        raise RuntimeError(f"Generated {len(outputs)} responses for {len(tasks)} tasks.")
    return outputs


# Model and SAE intervention

def load_model_and_tokenizer(
    model_id: str,
    dtype_name: str,
    trust_remote_code: bool,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch_dtype_from_name(dtype_name)
    if device.type == "cpu" and torch_dtype != torch.float32:
        raise ValueError(
            "CPU generation requires --dtype float32. Use a CUDA device for "
            f"dtype={dtype_name}."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model


def transformer_layer(model, layer: int):
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise TypeError(
            "Expected a Llama-style model exposing model.model.layers."
        ) from exc

    if layer < 0 or layer >= len(layers):
        raise ValueError(
            f"Layer {layer} is outside model range [0, {len(layers) - 1}]."
        )
    return layers[layer]


def apply_selected_sae_activation(
    sae,
    pre_activations: torch.Tensor,
    disabled_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply the SAE nonlinearity to selected features only."""
    activation = sae.act

    # LlamaScope JumpReLU stores one threshold per SAE feature. Calling the
    # full activation module on a reduced feature tensor would misalign those
    # thresholds, so select the matching thresholds explicitly.
    if hasattr(activation, "get_jumprelu_threshold"):
        thresholds = activation.get_jumprelu_threshold().reshape(-1)
        thresholds = thresholds.index_select(0, disabled_ids).to(
            device=pre_activations.device,
            dtype=pre_activations.dtype,
        )
        return pre_activations * (pre_activations > thresholds)

    if hasattr(activation, "log_jumprelu_threshold"):
        thresholds = activation.log_jumprelu_threshold.exp().reshape(-1)
        thresholds = thresholds.index_select(0, disabled_ids).to(
            device=pre_activations.device,
            dtype=pre_activations.dtype,
        )
        return pre_activations * (pre_activations > thresholds)

    # ReLU and other elementwise activations do not need feature-specific
    # parameters and can safely operate on the selected tensor.
    activated = activation(pre_activations)
    if activated.shape != pre_activations.shape:
        raise ValueError(
            "SAE activation changed the shape of selected feature activations: "
            f"{tuple(pre_activations.shape)} -> {tuple(activated.shape)}."
        )
    return activated


def build_sae_disable_hook(sae, disabled_feature_ids: Sequence[int]):
    feature_ids = sorted(set(int(feature) for feature in disabled_feature_ids))
    if not feature_ids:
        raise ValueError("At least one SAE feature must be disabled.")

    try:
        parameter = next(sae.parameters())
    except StopIteration as exc:
        raise TypeError("The loaded SAE has no parameters.") from exc

    device = parameter.device
    dtype = parameter.dtype

    if not hasattr(sae, "enc") or not isinstance(sae.enc, torch.nn.Linear):
        raise TypeError("Selective intervention requires sae.enc to be nn.Linear.")
    if not hasattr(sae, "dec") or not isinstance(sae.dec, torch.nn.Linear):
        raise TypeError("Selective intervention requires sae.dec to be nn.Linear.")
    if not hasattr(sae, "act"):
        raise TypeError("Selective intervention requires the SAE activation sae.act.")

    d_sae = int(sae.enc.weight.shape[0])
    invalid = [feature for feature in feature_ids if feature < 0 or feature >= d_sae]
    if invalid:
        raise ValueError(
            f"Feature ids outside SAE range [0, {d_sae - 1}]: {invalid}"
        )

    disabled_ids = torch.tensor(feature_ids, device=device, dtype=torch.long)

    # Encoder rows for the selected features only.
    encoder_weight = sae.enc.weight.index_select(0, disabled_ids).to(
        device=device,
        dtype=dtype,
    )
    encoder_bias = None
    if sae.enc.bias is not None:
        encoder_bias = sae.enc.bias.index_select(0, disabled_ids).to(
            device=device,
            dtype=dtype,
        )

    # nn.Linear(d_sae -> d_model) stores decoder vectors as columns.
    decoder_weight = sae.dec.weight.index_select(1, disabled_ids).to(
        device=device,
        dtype=dtype,
    )

    def hook_fn(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        if hidden_states is None:
            return output

        selected_pre_activations = F.linear(
            hidden_states,
            encoder_weight,
            encoder_bias,
        )
        selected_features = apply_selected_sae_activation(
            sae,
            selected_pre_activations,
            disabled_ids,
        )
        selected_contribution = selected_features @ decoder_weight.T
        modified_hidden_states = hidden_states - selected_contribution

        if isinstance(output, tuple):
            return (modified_hidden_states, *output[1:])
        if isinstance(output, list):
            return [modified_hidden_states, *output[1:]]
        return modified_hidden_states

    return hook_fn


@contextmanager
def sae_disabled_context(
    model,
    sae,
    layer: int,
    feature_ids: Sequence[int],
):
    hook = build_sae_disable_hook(sae, feature_ids)
    handle = transformer_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_responses(
    model,
    tokenizer,
    sae,
    layer: int,
    tasks: Sequence[Tuple[str, Optional[str], str, str]],
    disabled_feature_ids: Sequence[int],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    description: str,
) -> List[str]:
    device = next(model.parameters()).device
    set_seed(seed)

    outputs: List[str] = []
    do_sample = temperature > 0

    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            temperature=temperature,
            top_p=top_p,
        )

    with sae_disabled_context(model, sae, layer, disabled_feature_ids):
        for start in tqdm(
            range(0, len(tasks), batch_size),
            desc=description,
            unit="batch",
        ):
            batch = tasks[start : start + batch_size]
            prompts = [
                chat_text(tokenizer, system_prompt, question)
                for _, _, question, system_prompt in batch
            ]

            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_width = int(encoded["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    **generation_kwargs,
                )

            generated_only = generated[:, input_width:]
            batch_outputs = tokenizer.batch_decode(
                generated_only,
                skip_special_tokens=True,
            )
            outputs.extend(text.strip() for text in batch_outputs)

    if len(outputs) != len(tasks):
        raise RuntimeError(
            f"Generated {len(outputs)} responses for {len(tasks)} tasks."
        )

    return outputs


# Output planning and writing

def output_path_for_experiment(
    out_dir: Path,
    model_id: str,
    topic: str,
    person: str,
    country: str,
    mode: str,
    layer_spec: LayerSpec,
    expansion: str,
) -> Path:
    # Mirrors generate-responses.py's one-folder-per-mode convention.
    mode_dir = out_dir.parent / f"{out_dir.name}-{mode}"
    filename = "_".join(
        [
            MODEL,
            topic,
            person.replace(" ", ""),
            slugify(country),
            mode,
            layer_spec.layer_id,
            expansion,
            "sae_off",
        ]
    ) + ".csv"
    return mode_dir / filename


def build_experiment_plans(
    args: argparse.Namespace,
    country: str,
    layer_specs: Sequence[LayerSpec],
) -> List[ExperimentPlan]:
    plans: List[ExperimentPlan] = []

    for layer_spec in layer_specs:
        feature_file = salient_features_path(
            args.features_dir,
            layer_spec,
            args.expansion,
        )
        features_by_topic = load_layer_feature_rows(
            path=feature_file,
            country=country,
            topics=args.topics,
            num_features=args.num_features,
        )

        for topic in args.topics:
            for mode in args.modes:
                output_file = output_path_for_experiment(
                    out_dir=args.out_dir,
                    model_id=args.model_id,
                    topic=topic,
                    person=args.person,
                    country=country,
                    mode=mode,
                    layer_spec=layer_spec,
                    expansion=args.expansion,
                )
                plans.append(
                    ExperimentPlan(
                        layer_spec=layer_spec,
                        topic=topic,
                        mode=mode,
                        feature_ids=features_by_topic[topic],
                        features_file=feature_file,
                        output_file=output_file,
                    )
                )

    return plans


def write_generation_csv(
    path: Path,
    tasks: Sequence[Tuple[str, Optional[str], str, str]],
    outputs: Sequence[str],
) -> None:
    if len(tasks) != len(outputs):
        raise ValueError(
            f"Cannot write {len(outputs)} outputs for {len(tasks)} tasks."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for (country, name, question, system_prompt), output in zip(tasks, outputs):
            writer.writerow(
                {
                    "country": country,
                    "sys_prompt": system_prompt,
                    "name": name or "",
                    "question": question,
                    "output": output,
                }
            )


def write_metadata(
    plan: ExperimentPlan,
    args: argparse.Namespace,
    country: str,
    n_tasks: int,
) -> Path:
    metadata_path = Path(str(plan.output_file) + ".meta.json")
    metadata = {
        "model_id": args.model_id,
        "model_short": MODEL,
        "country": country,
        "topic": plan.topic,
        "mode": plan.mode,
        "person": args.person,
        "condition": "sae_off",
        "sae": {
            "layer": plan.layer_spec.layer,
            "site": plan.layer_spec.site,
            "layer_id": plan.layer_spec.layer_id,
            "expansion": args.expansion,
            "disabled_feature_ids": list(plan.feature_ids),
            "n_disabled_features": len(plan.feature_ids),
            "features_file": str(plan.features_file.resolve()),
            "features_grouping": "topic_country",
        },
        "generation": {
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "dtype": args.dtype,
            "n_tasks": n_tasks,
        },
        "output_columns": CSV_COLUMNS,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata_path


def print_plan_summary(plans: Sequence[ExperimentPlan]) -> None:
    print(f"[info] Planned experiments: {len(plans)}")
    for plan in plans:
        print(
            "[plan] "
            f"{plan.layer_spec.layer_id}/{plan.topic}/{plan.mode}: "
            f"{len(plan.feature_ids)} features -> {plan.output_file}"
        )


def multilayer_output_path(
    out_dir: Path,
    model_id: str,
    topic: str,
    person: str,
    country: str,
    mode: str,
    layer_features: Sequence[Tuple[LayerSpec, Sequence[int]]],
    expansion: str,
    n_features_per_layer: int,
) -> Path:
    mode_dir = out_dir / mode
    layer_tag = "-".join(layer.layer_id for layer, _features in layer_features)
    filename = "_".join(
        [
            MODEL,
            topic,
            person.replace(" ", ""),
            slugify(country),
            mode,
            "multilayer",
            "per_layer",
            layer_tag,
            expansion,
            f"top{n_features_per_layer}",
            "sae_off",
        ]
    ) + ".csv"
    return mode_dir / filename


def build_multilayer_plans(
    args: argparse.Namespace,
    country: str,
) -> List[MultiLayerExperimentPlan]:
    layer_specs = unique_preserving_order(
        parse_layer_spec(raw, default_site=args.site)
        for raw in args.layers
    )
    features_by_topic: Dict[str, List[Tuple[LayerSpec, Tuple[int, ...]]]] = {
        topic: [] for topic in args.topics
    }
    feature_files_by_topic: Dict[str, List[Tuple[LayerSpec, Path]]] = {
        topic: [] for topic in args.topics
    }

    for layer_spec in layer_specs:
        feature_file = salient_features_path(
            args.features_dir,
            layer_spec,
            args.expansion,
        )
        layer_rows = load_layer_feature_rows(
            path=feature_file,
            country=country,
            topics=args.topics,
            num_features=args.num_features,
            skip_missing=True,
        )

        for topic, feature_ids in layer_rows.items():
            if len(feature_ids) != args.num_features:
                raise ValueError(
                    f"Expected {args.num_features} features for "
                    f"{layer_spec.layer_id}/{topic}/{country}, "
                    f"found {len(feature_ids)} in {feature_file}."
                )
            features_by_topic[topic].append((layer_spec, feature_ids))
            feature_files_by_topic[topic].append((layer_spec, feature_file))

    plans: List[MultiLayerExperimentPlan] = []
    for topic in args.topics:
        layer_features = tuple(features_by_topic[topic])
        if not layer_features:
            print(
                f"[skip] No available SAE features for {topic}/{country} "
                "in any configured layer; omitting all modes for this topic."
            )
            continue
        for mode in args.modes:
            plans.append(
                MultiLayerExperimentPlan(
                    topic=topic,
                    mode=mode,
                    layer_features=layer_features,
                    feature_files=tuple(feature_files_by_topic[topic]),
                    output_file=multilayer_output_path(
                        out_dir=args.out_dir,
                        model_id=args.model_id,
                        topic=topic,
                        person=args.person,
                        country=country,
                        mode=mode,
                        layer_features=layer_features,
                        expansion=args.expansion,
                        n_features_per_layer=args.num_features,
                    ),
                )
            )
    return plans


def write_multilayer_metadata(
    plan: MultiLayerExperimentPlan,
    args: argparse.Namespace,
    country: str,
    n_tasks: int,
) -> Path:
    metadata_path = Path(str(plan.output_file) + ".meta.json")
    layer_map = {
        layer.layer_id: list(feature_ids)
        for layer, feature_ids in plan.layer_features
    }
    feature_file_map = {
        layer.layer_id: str(path.resolve())
        for layer, path in plan.feature_files
    }
    requested_layer_ids = unique_preserving_order(
        parse_layer_spec(raw, default_site=args.site).layer_id
        for raw in args.layers
    )
    metadata = {
        "model_id": args.model_id,
        "model_short": MODEL,
        "country": country,
        "topic": plan.topic,
        "mode": plan.mode,
        "person": args.person,
        "condition": "sae_off_multilayer",
        "sae": {
            "expansion": args.expansion,
            "disabled_features_by_layer": layer_map,
            "n_disabled_features": plan.n_features,
            "n_disabled_features_per_layer": args.num_features,
            "feature_files_by_layer": feature_file_map,
            "requested_layer_ids": requested_layer_ids,
            "skipped_layer_ids": [
                layer_id for layer_id in requested_layer_ids if layer_id not in layer_map
            ],
            "features_grouping": "topic_country_per_layer",
        },
        "generation": {
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "dtype": args.dtype,
            "n_tasks": n_tasks,
        },
        "output_columns": CSV_COLUMNS,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_path


def print_multilayer_plan_summary(plans: Sequence[MultiLayerExperimentPlan]) -> None:
    print(f"[info] Planned multilayer experiments: {len(plans)}")
    for plan in plans:
        distribution = ", ".join(
            f"{layer.layer_id}={len(ids)}"
            for layer, ids in plan.layer_features
        )
        print(
            f"[plan] multilayer/{plan.topic}/{plan.mode}: "
            f"{plan.n_features} total features ({distribution}) -> {plan.output_file}"
        )


def load_interventions_for_plan(
    plan: MultiLayerExperimentPlan,
    model,
    expansion: str,
) -> List[SelectedSAEIntervention]:
    interventions: List[SelectedSAEIntervention] = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for layer_spec, feature_ids in plan.layer_features:
        print(
            f"[info] Loading {layer_spec.layer_id} SAE to extract "
            f"{len(feature_ids)} selected features..."
        )
        sae, _hparams, _metadata = load_llamascope_sae(
            layer=layer_spec.layer,
            site=layer_spec.site,
            expansion=expansion,
            device=device,
            dtype=dtype,
        )
        sae.eval()
        intervention = extract_selected_intervention(
            sae=sae,
            layer_spec=layer_spec,
            feature_ids=feature_ids,
        )
        interventions.append(intervention)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return interventions


def main() -> None:
    args = parse_args()

    print("[info] Loading country/name data...")
    names_by_country = load_fb_names_json(
        args.fb_names_json,
        max_names_per_country=args.max_names_per_country,
    )
    if not names_by_country:
        raise RuntimeError(f"No country names were loaded from {args.fb_names_json}.")
    country = resolve_country_name(args.country, names_by_country)
    print(f"[info] Resolved country: {country}")

    print(f"[info] Reading per-layer feature files from: {args.features_dir}")
    plans = build_multilayer_plans(args, country)
    print_multilayer_plan_summary(plans)
    if not plans:
        print("[info] No experiments with available SAE features; nothing to generate.")
        return
    pending = [
        plan for plan in plans
        if args.overwrite or not plan.output_file.exists()
    ]
    for plan in plans:
        if plan not in pending:
            print(f"[skip] Output already exists: {plan.output_file}")
    if not pending:
        print("[info] No pending experiments. Use --overwrite to regenerate outputs.")
        return

    task_cache: Dict[Tuple[str, str], List[Tuple[str, Optional[str], str, str]]] = {}
    for plan in pending:
        task_cache[(plan.topic, plan.mode)] = build_tasks(
            mode=plan.mode,
            person=args.person,
            topic=plan.topic,
            country=country,
            names_by_country=names_by_country,
        )

    device = resolve_device(args.device)
    print(f"[info] Loading model={args.model_id}, device={device}, dtype={args.dtype}")
    tokenizer, model = load_model_and_tokenizer(
        model_id=args.model_id,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and (
        args.max_input_length + args.max_new_tokens > int(max_positions)
    ):
        raise ValueError(
            f"Configured lengths exceed context: {args.max_input_length} + "
            f"{args.max_new_tokens} > {max_positions}."
        )

    # Cache one lightweight intervention set per topic. Modes share the same features.
    intervention_cache: Dict[str, List[SelectedSAEIntervention]] = {}
    for topic in args.topics:
        topic_plans = [plan for plan in pending if plan.topic == topic]
        if not topic_plans:
            continue
        intervention_cache[topic] = load_interventions_for_plan(
            topic_plans[0], model=model, expansion=args.expansion
        )

    for plan in pending:
        tasks = task_cache[(plan.topic, plan.mode)]
        interventions = intervention_cache[plan.topic]
        description = f"multilayer|{plan.topic}|{plan.mode}|{country}|top{plan.n_features}"
        print(f"[info] Generating {description}")
        outputs = generate_responses_multilayer(
            model=model,
            tokenizer=tokenizer,
            interventions=interventions,
            tasks=tasks,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            description=description,
        )
        write_generation_csv(plan.output_file, tasks, outputs)
        metadata_path = write_multilayer_metadata(
            plan=plan, args=args, country=country, n_tasks=len(tasks)
        )
        print(f"[saved] {plan.output_file}")
        print(f"[saved] {metadata_path}")

    print("Multilayer generation completed.")


if __name__ == "__main__":
    main()
