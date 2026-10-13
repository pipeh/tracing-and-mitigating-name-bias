"""Build regression inputs from complete DCLM activations and judge labels.

No previous regression inputs are needed. Each layer must cover every name
and the union of CRISP features for the requested countries.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = None
JUDGE = 'llama-3.1-70b-w8a8'
MODEL = 'Meta-Llama-3.1-8B-Instruct'
LAYERS = [4, 8, 12, 16, 20, 24, 28]
COUNTRIES = ['India', 'Japan', 'Korea, Republic of', 'Russian Federation', 'Greece', 'Canada', 'Finland']
FACETS = {'Food': 'food', 'Clothing': 'clothing', 'Tradition': 'tradition', 'Rituals': 'rituals'}
THRESHOLD = 0.031329113990068436
MIN_TOKENS = 256
TOP_K = 10


def summarize(path, feature_ids, names):
    feature_ids = np.asarray(feature_ids, dtype=int)
    width = len(feature_ids)
    if not width or len(set(feature_ids)) != width or np.any(feature_ids < 0) or np.any(feature_ids >= 32768):
        raise ValueError('Expected unique feature IDs in the 8x SAE range [0, 32768).')
    if names.duplicated(['country', 'name']).any():
        raise ValueError('Duplicate country/name pairs.')
    lookup = np.full(32768, -1, dtype=int)
    lookup[feature_ids] = np.arange(width)
    name_lookup = {(row.country, row.name): i for i, row in enumerate(names.itertuples())}
    counts = np.zeros(len(names), dtype=int)
    active = np.zeros((len(names), width), dtype=int)
    top = np.zeros((len(names), TOP_K, width), dtype=np.float32)
    columns = ['country', 'name', 'n_tokens', 'token_feature_idx', 'token_feature_act']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        mask = pc.and_(pc.greater_equal(batch.column('n_tokens'), MIN_TOKENS),
                       pc.is_in(batch.column('country'), value_set=pa.array(COUNTRIES, type=batch.column('country').type)))
        batch = batch.filter(mask)
        if not batch.num_rows:
            continue
        nested_ids = batch.column('token_feature_idx')
        token_ids = pc.list_flatten(nested_ids)
        nested_values = batch.column('token_feature_act')
        token_values = pc.list_flatten(nested_values)
        lengths = batch.column('n_tokens').to_numpy()
        if (not np.array_equal(pc.list_value_length(nested_ids).to_numpy(), lengths)
                or not np.array_equal(pc.list_value_length(nested_values).to_numpy(), lengths)
                or not np.array_equal(pc.list_value_length(token_ids).to_numpy(),
                                      pc.list_value_length(token_values).to_numpy())):
            raise ValueError(f'Misaligned token activations in {path}.')
        ids = pc.list_flatten(token_ids).to_numpy()
        values = pc.list_flatten(token_values).to_numpy().astype(np.float32)
        if np.any(ids < 0) or np.any(ids >= len(lookup)) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f'Invalid activation coordinates in {path}.')
        slots = lookup[ids]
        keep = slots >= 0
        tokens = pc.list_parent_indices(token_ids).to_numpy()[keep]
        documents = pc.list_parent_indices(nested_ids).to_numpy()[tokens]
        sums = np.bincount(documents * width + slots[keep], weights=values[keep],
                           minlength=batch.num_rows * width).reshape(batch.num_rows, width)
        matrix = (sums / batch.column('n_tokens').to_numpy()[:, None]).astype(np.float32)
        rows = np.array([name_lookup[key] for key in zip(batch.column('country').to_pylist(), batch.column('name').to_pylist())])
        for row in np.unique(rows):
            block = matrix[rows == row]
            counts[row] += len(block)
            active[row] += (block > THRESHOLD).sum(axis=0)
            candidates = np.concatenate([top[row], block])
            top[row] = np.partition(candidates, -TOP_K, axis=0)[-TOP_K:]
    if np.any(counts == 0):
        missing = names.loc[counts == 0, ['country', 'name']].to_dict('records')
        raise ValueError(f'No qualifying documents in {path} for {missing}.')
    return (active / counts[:, None]).astype(np.float32), \
        (top.sum(axis=1) / counts.clip(max=TOP_K)[:, None]).astype(np.float32), counts


def additional_labels(country):
    rows, global_answers, global_baseline = {}, [], []
    judge_dir = DATA_DIR / f'analysis/{JUDGE}'
    for target, facet in FACETS.items():
        answers = pd.read_csv(judge_dir / f'with_name/{MODEL}/{facet}_FirstPerson_{country}_{country}.tsv',
                              sep='\t', usecols=['name', 'decision'])
        answers['positive'] = answers.decision.str.strip().str.lower().eq('yes').astype(float)
        baseline = pd.read_csv(judge_dir / f'woname/{MODEL}/{facet}_FirstPerson_baseline_{country}.tsv',
                               sep='\t', usecols=['decision']).decision.str.strip().str.lower().eq('yes')
        if answers.empty or baseline.empty:
            raise ValueError(f'Empty judge results for {country}/{facet}.')
        for name, value in answers.groupby('name').positive.mean().items():
            rows.setdefault(name, {'country': country, 'name': name})[target] = value - baseline.mean()
        global_answers.append(answers[['name', 'positive']])
        global_baseline.extend(baseline.tolist())
    for name, value in pd.concat(global_answers).groupby('name').positive.mean().items():
        rows[name]['Global'] = value - np.mean(global_baseline)
    return pd.DataFrame(rows.values())


def prepare(args):
    first = args.acts_dir / f'llamascope_salient_token_acts_dclm_L{LAYERS[0]}R_8x.parquet'
    metadata = json.loads(Path(str(first) + '.meta.json').read_text())
    names = pd.DataFrame([{'country': country, 'name': name} for country in COUNTRIES
                          for name in metadata['names_by_country'][country]])
    names = names.sort_values(['name', 'country']).reset_index(drop=True)
    if names.empty or names.duplicated(['country', 'name']).any():
        raise ValueError('Missing or duplicate names in activation metadata.')
    context_rows, feature_rows, frequencies, top_means = [], [], [], []
    n_docs = None
    offset = 0
    for layer in LAYERS:
        path = args.acts_dir / f'llamascope_salient_token_acts_dclm_L{layer}R_8x.parquet'
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
        for country in COUNTRIES:
            if sorted(meta['names_by_country'][country]) != sorted(metadata['names_by_country'][country]):
                raise ValueError(f'Name coverage differs at L{layer}: {country}.')
        ids = np.array(sorted(set(int(feature) for feature in meta['feature_ids'])))
        for directory in args.features_dirs:
            feature_path = directory / f'topic_country_salient_features_{MODEL}_L{layer}R_8x.parquet'
            for record in pq.read_table(feature_path, columns=['country', 'topic', 'salient_features']).to_pylist():
                if record['country'] not in COUNTRIES or record['topic'] not in FACETS.values():
                    continue
                for rank, feature in enumerate(record['salient_features'], 1):
                    if int(feature) not in ids:
                        raise ValueError(f'CRISP feature {feature} absent from extracted L{layer}.')
                    context_rows.append({'country': record['country'], 'facet': record['topic'],
                                         'layer': layer, 'feature_id': int(feature), 'rank': rank})
        freq, topk, counts = summarize(path, ids, names)
        if n_docs is None:
            n_docs = counts
        np.testing.assert_array_equal(counts, n_docs, err_msg=f'Document counts differ at L{layer}.')
        feature_rows.extend({'vector_index': offset + i, 'layer': layer, 'feature_id': int(feature)}
                            for i, feature in enumerate(ids))
        offset += len(ids)
        frequencies.append(freq)
        top_means.append(topk)
    contexts = pd.DataFrame(context_rows).drop_duplicates()
    if contexts.empty:
        raise ValueError('No matching CRISP contexts.')
    if contexts.duplicated(['country', 'facet', 'layer', 'rank']).any():
        raise ValueError('Conflicting CRISP rankings in --features-dirs.')
    names['n_docs'] = n_docs
    labels = pd.concat([additional_labels(country) for country in COUNTRIES], ignore_index=True)
    names = names.merge(labels, on=['country', 'name'], how='left', validate='one_to_one')
    if names[['Global', *FACETS]].isna().any().any():
        raise ValueError('Missing bias labels for some names.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names.to_csv(args.output_dir / 'aligned_names.csv', index=False)
    pd.DataFrame(feature_rows).to_csv(args.output_dir / 'feature_map_all.csv', index=False)
    contexts.to_csv(args.output_dir / 'context_feature_provenance.csv', index=False)
    np.savez_compressed(args.output_dir / 'name_activation_summaries.npz',
                        freq_global=np.concatenate(frequencies, axis=1),
                        topk_mean=np.concatenate(top_means, axis=1))
    (args.output_dir / 'run.json').write_text(json.dumps({
        'countries': COUNTRIES, 'layers': LAYERS, 'min_tokens': MIN_TOKENS,
        'topk_mean_documents': TOP_K, 'frequency_threshold': THRESHOLD,
        'n_names': len(names), 'n_features': offset, 'activations_dir': str(args.acts_dir),
        'features_dirs': [str(path) for path in args.features_dirs],
        'judge': JUDGE, 'model': MODEL,
    }, indent=2) + '\n')
    print(f'Saved {len(names)} names and {offset} features to {args.output_dir}')


def main():
    global DATA_DIR, JUDGE, MODEL, COUNTRIES, LAYERS, MIN_TOKENS, TOP_K, THRESHOLD
    env = dotenv_values(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=env.get('DATA_DIR'))
    parser.add_argument('--judge', default=env.get('JUDGE_MODEL_ID', JUDGE))
    parser.add_argument('--model', default=env.get('GENERATION_MODEL_SHORT', MODEL))
    parser.add_argument('--countries', nargs='+', default=COUNTRIES)
    parser.add_argument('--layers', type=int, nargs='+', default=LAYERS)
    parser.add_argument('--acts-dir', type=Path)
    parser.add_argument('--features-dirs', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--min-tokens', type=int, default=MIN_TOKENS)
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error('Set DATA_DIR in .env or pass --data-dir.')
    if args.min_tokens < 1 or args.top_k < 1 or args.threshold < 0:
        parser.error('Invalid token cutoff, top-k, or threshold.')
    if len(set(args.layers)) != len(args.layers) or len(set(args.countries)) != len(args.countries):
        parser.error('Countries and layers must be unique.')
    DATA_DIR = args.data_dir.expanduser().resolve()
    JUDGE, MODEL, COUNTRIES, LAYERS = args.judge, args.model, args.countries, args.layers
    MIN_TOKENS, TOP_K, THRESHOLD = args.min_tokens, args.top_k, args.threshold
    args.acts_dir = args.acts_dir or DATA_DIR / 'sae-salient-tokens' / JUDGE
    args.features_dirs = args.features_dirs or [DATA_DIR / 'crisp' / JUDGE]
    args.output_dir = args.output_dir or DATA_DIR / 'analysis/sae-regression-seven-countries/inputs'
    prepare(args)


if __name__ == '__main__':
    main()
