"""Freeze random SAE controls, name selections, questions, and prompts.

For each country/facet/layer, exclude the 20 CRISP features and sample 20
from the most frequent 10% of remaining features for three seeds. Frequency
counts positive token activations across both bias labels; ties use feature
ID. Sample 20 names/country and 25% of questions/facet. Store selections
and generation settings in experiment.json."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq

GENERATOR_PATH = Path(__file__).with_name("generate-responses-sae-off-random.py")
spec = importlib.util.spec_from_file_location("random_sae_generator", GENERATOR_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {GENERATOR_PATH}")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
base = generator.base

from crisp_utils import infer_topic_from_source_file  # noqa: E402
from generation_utils import get_questions  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", default=generator.COUNTRIES, help="Exact country names from the data.")
    parser.add_argument("--topics", nargs="+", choices=generator.TOPICS, help="Restrict preparation to these facets.")
    parser.add_argument("--features_dir", type=Path, default=base.DEFAULT_FEATURES_DIR)
    parser.add_argument("--activations_dir", type=Path, help="Override activation paths from CRISP metadata.")
    parser.add_argument("--out_dir", type=Path, default=generator.DEFAULT_SELECTION_DIR)
    parser.add_argument("--generations_dir", type=Path, default=base.DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--name_seed", type=int, default=0)
    parser.add_argument("--question_seed", type=int, default=0)
    parser.add_argument("--pool_fraction", type=float, default=0.10)
    parser.add_argument("--batch_size", type=int, default=4, help="Documents per CPU activation-reading batch.")
    args = parser.parse_args()
    if len(set(args.seeds)) != 3:
        parser.error("--seeds must contain three distinct seeds.")
    if not 0 < args.pool_fraction <= 1:
        parser.error("--pool_fraction must be in (0, 1].")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    return args


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sample_inputs(args, config):
    available = base.load_fb_names_json(Path(config["fb_names_json"]), config["max_names_per_country"])
    names = {}
    for country in config["topics_by_country"]:
        if country not in available:
            raise ValueError(f"Country name absent from the name data: {country}")
        candidates = available[country]
        if len(candidates) < generator.N_NAMES:
            raise ValueError(f"Expected at least 20 names for {country}.")
        rng = random.Random(f"names:{args.name_seed}:{country}")
        indices = sorted(rng.sample(range(len(candidates)), generator.N_NAMES))
        names[country] = {
            "all_names": candidates, "selected_indices": indices,
            "names": [candidates[index] for index in indices],
        }
    questions = {}
    for topic in generator.TOPICS:
        candidates = get_questions(topic)
        if not candidates:
            raise ValueError(f"No questions available for {topic}.")
        rng = random.Random(f"questions:{args.question_seed}:{topic}")
        indices = sorted(rng.sample(range(len(candidates)), math.ceil(len(candidates) * 0.25)))
        questions[topic] = {
            "all_questions": candidates, "selected_indices": indices,
            "questions": [candidates[index] for index in indices],
        }
    tasks = {}
    selected_names = {country: value["names"] for country, value in names.items()}
    for country in selected_names:
        tasks[country] = {}
        for topic in config["topics_by_country"][country]:
            selection = questions[topic]
            indices = set(selection["selected_indices"])
            n_questions = len(selection["all_questions"])
            tasks[country][topic] = {
                mode: [list(task) for i, task in enumerate(base.build_tasks(
                    mode, config["person"], topic, country, selected_names,
                )) if i % n_questions in indices]
                for mode in config["modes"]
            }
    return names, questions, tasks


def count_activations(path, d_sae, batch_size, topics_by_country):
    """Count the stored positive coordinates, combining all documents/labels."""
    counts = {
        (country, topic): np.zeros(d_sae, dtype=np.int64)
        for country, topics in topics_by_country.items() for topic in topics
    }
    documents = {key: 0 for key in counts}
    tokens = {key: 0 for key in counts}
    parquet = pq.ParquetFile(path)
    columns = ["country", "source_file", "n_tokens", "token_feature_idx"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        countries = batch.column("country").to_pylist()
        sources = batch.column("source_file").to_pylist()
        lengths = batch.column("n_tokens").to_pylist()
        features = batch.column("token_feature_idx")
        for i, country in enumerate(countries):
            if country not in topics_by_country:
                continue
            topic = infer_topic_from_source_file(sources[i])
            key = (country, topic)
            if key not in counts:
                continue
            # The original extractor stores one coordinate per positive
            # (token, feature), so a histogram counts activating tokens.
            per_token = features[i].values
            if len(per_token) != lengths[i]:
                raise ValueError(f"Token-count mismatch in {path}, {country}/{topic}.")
            ids = per_token.flatten().to_numpy(zero_copy_only=False)
            if ids.size and (ids.min() < 0 or ids.max() >= d_sae):
                raise ValueError(f"Feature outside SAE range in {path}.")
            counts[key] += np.bincount(ids, minlength=d_sae)
            documents[key] += 1
            tokens[key] += int(lengths[i])
    for key in counts:
        if not documents[key] or not tokens[key]:
            raise ValueError(f"Missing activation documents/tokens for {key} in {path}.")
    return counts, documents, tokens


def candidate_pool(counts, excluded, fraction):
    eligible = np.flatnonzero(counts > 0)
    eligible = eligible[~np.isin(eligible, excluded)]
    ranked = eligible[np.lexsort((eligible, -counts[eligible]))]
    n_candidates = math.ceil(len(ranked) * fraction)
    if n_candidates < generator.N_FEATURES:
        raise ValueError(f"Only {n_candidates} candidates in the frequent-feature pool; need 20.")
    return ranked[:n_candidates].tolist()


def prepare_layer(args, config, layer):
    feature_file = base.salient_features_path(args.features_dir, layer, config["expansion"])
    crisp_metadata = generator.read_json(Path(str(feature_file) + ".meta.json"))
    activation_file = (
        args.activations_dir / f"{base.MODEL}_{layer.layer_id}_{config['expansion']}.parquet"
        if args.activations_dir else Path(crisp_metadata["input_file"])
    )
    activation_meta_file = Path(str(activation_file) + ".meta.json")
    activation_metadata = generator.read_json(activation_meta_file)
    if activation_metadata.get("activation_scope") != "all_positive_per_token":
        raise ValueError(f"Expected all_positive_per_token activations: {activation_meta_file}")
    for field, expected in (("layer", layer.layer), ("site", layer.site), ("expansion", config["expansion"])):
        if activation_metadata[field] != expected:
            raise ValueError(f"Activation {field} mismatch in {activation_meta_file}.")
    d_sae = int(activation_metadata["d_sae"])
    originals = {
        country: base.load_layer_feature_rows(
            feature_file, country, config["topics_by_country"][country], generator.N_FEATURES,
        ) for country in config["topics_by_country"]
    }
    for country, topics in originals.items():
        for topic, ids in topics.items():
            if len(ids) != generator.N_FEATURES or any(i < 0 or i >= d_sae for i in ids):
                raise ValueError(f"Invalid original features: {layer.layer_id}/{country}/{topic}")
    inputs = {
        "activations": fingerprint(activation_file), "activation_metadata": fingerprint(activation_meta_file),
        "original_features": fingerprint(feature_file), "pool_fraction": args.pool_fraction,
        "d_sae": d_sae, "countries": list(config["topics_by_country"]),
        "topics_by_country": config["topics_by_country"],
    }
    cache_path = args.out_dir / "feature_pools" / f"{layer.layer_id}.json"
    if cache_path.exists():
        cached = generator.read_json(cache_path)
        if cached["inputs"] != inputs:
            raise ValueError(f"Candidate pool inputs changed: {cache_path}. Use a new selection directory.")
        print(f"[cached] {layer.layer_id}: {cache_path}")
        return cached
    print(f"[info] Counting positive activations for {layer.layer_id}: {activation_file}", flush=True)
    counts, documents, tokens = count_activations(activation_file, d_sae, args.batch_size, config["topics_by_country"])
    groups = {}
    for country in config["topics_by_country"]:
        groups[country] = {}
        for topic in config["topics_by_country"][country]:
            key = (country, topic)
            ids = candidate_pool(counts[key], originals[country][topic], args.pool_fraction)
            groups[country][topic] = {
                "candidate_ids": ids,
                "activation_counts": [int(counts[key][feature]) for feature in ids],
                "excluded_ids": list(originals[country][topic]),
                "n_documents": documents[key], "n_tokens": tokens[key],
            }
    cached = {"inputs": inputs, "groups": groups}
    generator.save_json(cache_path, cached)
    return cached


def prepare(args):
    config = generator.default_config()
    config["topics_by_country"] = {
        country: [topic for topic in generator.topics_for_country(country)
                  if args.topics is None or topic in args.topics]
        for country in args.countries
    }
    config["generations_dir"] = str(args.generations_dir)
    names, questions, tasks = sample_inputs(args, config)
    pools = {}
    feature_files = {}
    for raw in config["layers"]:
        layer = base.parse_layer_spec(raw, config["site"])
        pools[layer.layer_id] = prepare_layer(args, config, layer)
        feature_files[layer.layer_id] = str(base.salient_features_path(
            args.features_dir, layer, config["expansion"],
        ).resolve())
    runs = []
    for country in config["topics_by_country"]:
        for topic in config["topics_by_country"][country]:
            for seed in args.seeds:
                features = {}
                excluded = {}
                for layer_id, pool in pools.items():
                    group = pool["groups"][country][topic]
                    rng = random.Random(f"{seed}:{country}:{topic}:{layer_id}")
                    features[layer_id] = sorted(rng.sample(group["candidate_ids"], generator.N_FEATURES))
                    excluded[layer_id] = group["excluded_ids"]
                runs.append({
                    "country": country, "topic": topic, "seed": seed,
                    "layer_features": features, "excluded_features_by_layer": excluded,
                    "feature_files_by_layer": feature_files,
                })
    prepared = {
        "schema_version": 2, "config": config, "countries": list(config["topics_by_country"]),
        "seeds": list(args.seeds),
        "name_sampling": {"seed": args.name_seed, "n_names": generator.N_NAMES, "countries": names},
        "question_sampling": {"seed": args.question_seed, "fraction": 0.25, "rounding": "ceil", "facets": questions},
        "feature_sampling": {
            "criterion": "positive_token_frequency", "labels": "ignored",
            "pool_fraction": args.pool_fraction, "rounding": "ceil",
            "pool_definition": "top fraction of observed features after excluding original features",
            "tie_breaker": "feature_id_ascending", "sampling": "uniform_without_replacement",
            "rng": "random.Random(f'{seed}:{country}:{topic}:{layer_id}')",
            "pool_files": {layer_id: str((args.out_dir / "feature_pools" / f"{layer_id}.json").resolve()) for layer_id in pools},
        },
        "tasks": tasks, "runs": runs,
    }
    path = args.out_dir / "experiment.json"
    generator.save_json(path, prepared)
    n_tasks = sum(len(tasks[country][topic][mode]) for country in tasks for topic in tasks[country] for mode in config["modes"])
    print(f"[saved] {path}: {len(runs)} selections; {n_tasks} responses per seed")
    return prepared


if __name__ == "__main__":
    prepare(parse_args())
