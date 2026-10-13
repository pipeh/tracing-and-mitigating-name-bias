"""Generate random SAE controls from frozen experiment.json selections.

Reuses the multilayer generator without resampling. CSVs and sidecars are
written under DATA_DIR/generations-sae-off/<judge>/."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


BASE_PATH = Path(__file__).with_name("generate-responses-sae-off-multilayer.py")
spec = importlib.util.spec_from_file_location("sae_off_multilayer", BASE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load the original generator: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

COUNTRIES = ("India", "Japan", "Korea, Republic of", "Russian Federation", "Finland")
TOPICS = ("food", "clothing", "tradition", "rituals")
LAYERS = (4, 8, 12, 16, 20, 24, 28)
N_FEATURES = 20
N_NAMES = 20
DEFAULT_SELECTION_DIR = base.DATA_DIR / "analysis-sae-off" / "random-ablations"


def topics_for_country(country):
    """Finland uses only the three facets with original CRISP selections."""
    return tuple(topic for topic in TOPICS if country != "Finland" or topic != "food")


def default_config():
    """Freeze the original experimental defaults at preparation time."""
    return {
        "layers": [str(layer) for layer in LAYERS],
        "topics": list(TOPICS), "num_features": N_FEATURES,
        "topics_by_country": {country: list(topics_for_country(country)) for country in COUNTRIES},
        "site": base.DEFAULT_SITE, "expansion": base.DEFAULT_EXPANSION,
        "modes": list(base.DEFAULT_MODES), "person": base.DEFAULT_PERSON,
        "model_id": base.DEFAULT_MODEL_ID,
        "fb_names_json": str(Path(base.ROOT) / base.DEFAULT_FB_NAMES_JSON),
        "max_names_per_country": base.DEFAULT_MAX_NAMES_PER_COUNTRY,
        "batch_size": base.DEFAULT_BATCH_SIZE,
        "max_input_length": base.DEFAULT_MAX_INPUT_LENGTH,
        "max_new_tokens": base.DEFAULT_MAX_NEW_TOKENS,
        "temperature": base.DEFAULT_TEMPERATURE, "top_p": base.DEFAULT_TOP_P,
        "seed": base.DEFAULT_SEED, "dtype": base.DEFAULT_DTYPE,
        "trust_remote_code": True,
        "generations_dir": str(base.DEFAULT_OUT_DIR),
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    """Atomically persist selections; refuse to replace a different experiment."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Saved inputs differ from this experiment: {path}. Use a new selection directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument(
        "--countries", nargs="+",
        help='Exact country names from the data, e.g. "Korea, Republic of" "Russian Federation" Finland.',
    )
    parser.add_argument("--run_seeds", type=int, nargs="+", help="Prepared seeds to execute (default: all).")
    parser.add_argument("--generations_dir", type=Path, help="Override the prepared CSV output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.run_seeds is not None and len(set(args.run_seeds)) != len(args.run_seeds):
        parser.error("--run_seeds cannot contain duplicates.")
    return args


def load_prepared(args):
    path = args.selection_dir / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare-random-sae-ablations.py first; missing {path}")
    content = path.read_bytes()
    prepared = json.loads(content)
    if prepared.get("schema_version") != 2:
        raise ValueError(f"Unsupported prepared experiment format: {path}. Prepare a new selection directory.")
    countries = set(args.countries if args.countries is not None else prepared["countries"])
    unknown = countries - set(prepared["countries"])
    if unknown:
        raise ValueError(
            f"Unprepared country names: {sorted(unknown)}. Use the exact data names: {prepared['countries']}."
        )
    seeds = args.run_seeds if args.run_seeds is not None else prepared["seeds"]
    if not set(seeds) <= set(prepared["seeds"]):
        raise ValueError(f"--run_seeds must be selected from {prepared['seeds']}.")
    config = argparse.Namespace(**prepared["config"])
    config.generations_dir = args.generations_dir or Path(config.generations_dir)
    config.overwrite = args.overwrite
    selected = [run for run in prepared["runs"] if run["country"] in countries and run["seed"] in seeds]
    return prepared, config, selected, hashlib.sha256(content).hexdigest()


def plans_for_run(run, config):
    layer_features = tuple(
        (base.parse_layer_spec(raw, config.site), tuple(run["layer_features"][f"L{raw}R"]))
        for raw in config.layers
    )
    for layer, ids in layer_features:
        if len(ids) != N_FEATURES or len(set(ids)) != N_FEATURES:
            raise ValueError(f"Expected 20 unique prepared features in {layer.layer_id}.")
        if set(ids) & set(run["excluded_features_by_layer"][layer.layer_id]):
            raise ValueError(f"Prepared features overlap original features in {layer.layer_id}.")
    plans = []
    for mode in config.modes:
        path = base.multilayer_output_path(
            config.generations_dir, config.model_id, run["topic"], config.person,
            run["country"], mode, layer_features, config.expansion, config.num_features,
        )
        path = path.with_name(f"{path.stem}_random_ablation_{run['seed']}.csv")
        plans.append(base.MultiLayerExperimentPlan(
            topic=run["topic"], mode=mode, layer_features=layer_features,
            feature_files=tuple(
                (layer, Path(run["feature_files_by_layer"][layer.layer_id])) for layer, _ids in layer_features
            ), output_file=path,
        ))
    return plans


def pending_plans(plans, config, manifest_hash):
    pending = []
    for plan in plans:
        metadata_path = Path(str(plan.output_file) + ".meta.json")
        if config.overwrite or not plan.output_file.exists() or not metadata_path.exists():
            pending.append(plan)
        elif read_json(metadata_path).get("prepared_experiment", {}).get("sha256") != manifest_hash:
            raise ValueError(f"Existing output belongs to different selections: {plan.output_file}")
    return pending


@contextmanager
def run_lock(directory, run):
    """Avoid duplicate generation if workers accidentally select the same run."""
    path = directory / ".random-ablation-locks" / (
        f"{base.slugify(run['country'])}_{run['topic']}_{run['seed']}.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)


def save_generation(plan, run, tasks, outputs, config, prepared, selection_dir, manifest_hash):
    temporary_csv = Path(str(plan.output_file) + f".{os.getpid()}.tmp")
    base.write_generation_csv(temporary_csv, tasks, outputs)
    metadata_path = base.write_multilayer_metadata(plan, config, run["country"], len(tasks))
    metadata = read_json(metadata_path)
    metadata["condition"] = "sae_off_multilayer_random"
    metadata["sae"]["selection_seed"] = run["seed"]
    metadata["sae"]["feature_sampling"] = prepared["feature_sampling"]
    metadata["name_sampling"] = {
        "seed": prepared["name_sampling"]["seed"], "n_names": N_NAMES,
        "names": prepared["name_sampling"]["countries"][run["country"]]["names"],
    }
    metadata["question_sampling"] = {
        "seed": prepared["question_sampling"]["seed"], "fraction": 0.25,
        **prepared["question_sampling"]["facets"][run["topic"]],
    }
    metadata["prepared_experiment"] = {
        "file": str((selection_dir / "experiment.json").resolve()), "sha256": manifest_hash,
    }
    temporary_meta = Path(str(metadata_path) + f".{os.getpid()}.tmp")
    temporary_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_meta.replace(metadata_path)
    temporary_csv.replace(plan.output_file)


def main():
    args = parse_args()
    prepared, config, runs, manifest_hash = load_prepared(args)
    work = []
    for run in runs:
        pending = pending_plans(plans_for_run(run, config), config, manifest_hash)
        print(f"[plan] {run['country']}/{run['topic']}/seed_{run['seed']}: {len(pending)} pending CSVs")
        if pending:
            work.append(run)
    if args.dry_run or not work:
        print("[info] No generations executed.")
        return
    tokenizer, model = base.load_model_and_tokenizer(
        config.model_id, config.dtype, config.trust_remote_code, base.resolve_device(args.device)
    )
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and config.max_input_length + config.max_new_tokens > max_positions:
        raise ValueError("Configured input/output lengths exceed the model context.")
    for run in work:
        with run_lock(config.generations_dir, run) as acquired:
            if not acquired:
                print(f"[skip] Another worker is processing {run['country']}/{run['topic']}/{run['seed']}")
                continue
            pending = pending_plans(plans_for_run(run, config), config, manifest_hash)
            if not pending:
                continue
            interventions = base.load_interventions_for_plan(pending[0], model, config.expansion)
            for plan in pending:
                tasks = prepared["tasks"][run["country"]][run["topic"]][plan.mode]
                outputs = base.generate_responses_multilayer(
                    model=model, tokenizer=tokenizer, interventions=interventions, tasks=tasks,
                    batch_size=config.batch_size, max_input_length=config.max_input_length,
                    max_new_tokens=config.max_new_tokens, temperature=config.temperature,
                    top_p=config.top_p, seed=config.seed,
                    description=f"random|{run['country']}|{plan.topic}|seed_{run['seed']}|{plan.mode}",
                )
                save_generation(
                    plan, run, tasks, outputs,
                    config, prepared, args.selection_dir, manifest_hash,
                )
                print(f"[saved] {plan.output_file}")
            del interventions


if __name__ == "__main__":
    main()
