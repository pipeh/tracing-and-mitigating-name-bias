#!/usr/bin/env python
"""MMLU test evaluation: baseline plus seven-layer SAE-off conditions.

Default: classic 5-shot prompt, demonstrations from dev of the same subject,
and selection by the next-token likelihood of A/B/C/D. --shots 0 enables
zero-shot. --prompt_format chat explicitly opts into a custom Instruct format.
All results/checkpoints/metrics are JSON in DATA_DIR/mmlu.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import inspect
import json
import math
from pathlib import Path
import random

import torch
from tqdm import tqdm
import eval_llama_gsm8k_multilayer_140_sae_off as sae_eval
from eval_llama_culturalbench_multilayer_140_sae_off import digest, write_json

DATASET = 'cais/mmlu'
REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'
CHOICES = 'ABCD'
PROMPT_VERSION = 'hendrycks_subject_dev_prompt_v1'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features_dir', type=Path, default=sae_eval.DEFAULT_FEATURES_DIR)
    parser.add_argument('--feature_model_name', default=sae_eval.ENV.get('GENERATION_MODEL_SHORT', 'Meta-Llama-3.1-8B-Instruct'))
    parser.add_argument('--country', action='append', help='Filter intervention countries only')
    parser.add_argument('--facet', action='append', help='Filter intervention facets only')
    parser.add_argument('--num_features_per_layer', type=int, default=20)
    parser.add_argument('--model_id', default=str(sae_eval.DEFAULT_MODEL))
    parser.add_argument('--expansion', choices=('8x', '32x'), default='8x')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='float16')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--shots', type=int, choices=range(6), default=5)
    parser.add_argument('--prompt_format', choices=('plain', 'chat'), default='plain')
    parser.add_argument('--subject', action='append', help='Filter MMLU subjects; repeat for several')
    parser.add_argument('--n_samples', type=int, help='N total questions, seeded shuffle after subject filtering')
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--max_input_length', type=int, default=8192)
    parser.add_argument('--output_dir', type=Path, help='Default: DATA_DIR/mmlu')
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--resume_partial', action='store_true')
    parser.add_argument('--list_conditions', action='store_true')
    parser.add_argument('--dataset_dir', type=Path, help='Offline folder containing all/test-*.parquet and all/dev-*.parquet')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = sae_eval.default_output_dir().with_name('mmlu')
    for field in ('num_features_per_layer', 'max_input_length', 'save_every', 'n_samples'):
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f'--{field} must be positive')
    return args


def normalize_rows(records, split):
    rows = []
    for i, record in enumerate(records):
        choices = list(record['choices'])
        answer = record['answer']
        if isinstance(answer, bool) or int(answer) != answer or not 0 <= int(answer) < 4 or len(choices) != 4:
            raise ValueError(f'Invalid MMLU label/choices at {split}[{i}]')
        question, subject = record['question'], record['subject']
        if not isinstance(question, str) or not question.strip() or not isinstance(subject, str) or not subject.strip():
            raise ValueError(f'Invalid question/subject at {split}[{i}]')
        if any(not isinstance(option, str) or not option.strip() for option in choices):
            raise ValueError(f'Invalid choice at {split}[{i}]')
        rows.append({'id': f'{split}:{i}', 'subject': subject, 'question': question,
                     'choices': choices, 'gold_answer': CHOICES[int(answer)]})
    if not rows:
        raise ValueError(f'Empty MMLU {split}')
    return rows


def prepare_data(test_records, dev_records, args):
    test, dev = normalize_rows(test_records, 'test'), normalize_rows(dev_records, 'dev') if args.shots else []
    subjects = {row['subject'] for row in test}
    missing = set(args.subject or []) - subjects
    if missing:
        raise ValueError(f'Unknown MMLU subjects: {sorted(missing)}')
    test = [row for row in test if not args.subject or row['subject'] in args.subject]
    test.sort(key=lambda row: (row['subject'], int(row['id'].split(':')[1])))
    if args.n_samples is not None:
        random.Random(args.sample_seed).shuffle(test)
        test = test[:args.n_samples]
    demonstrations = defaultdict(list)
    for row in dev:
        if len(demonstrations[row['subject']]) < args.shots:
            demonstrations[row['subject']].append(row)
    for subject in {row['subject'] for row in test}:
        if len(demonstrations[subject]) != args.shots:
            raise ValueError(f'Expected {args.shots} dev demonstrations for {subject}')
    return test, dict(demonstrations)


def format_example(row, include_answer=False):
    text = row['question'] + ''.join(f'\n{letter}. {option}' for letter, option in zip(CHOICES, row['choices'])) + '\nAnswer:'
    if include_answer:
        text += f' {row["gold_answer"]}\n\n'
    return text


def build_prompt(row, demonstrations):
    # Preserve the original implementation's extra space before subject name.
    subject = ' ' + row['subject'].replace('_', ' ')
    return (f'The following are multiple choice questions (with answers) about {subject}.\n\n'
            + ''.join(format_example(example, True) for example in demonstrations)
            + format_example(row))


def tokenize_prompt(tokenizer, row, demonstrations, args):
    prompt = build_prompt(row, demonstrations)
    if args.prompt_format == 'plain':
        if tokenizer.bos_token_id is None or not tokenizer.bos_token:
            raise ValueError('Plain Llama protocol requires a tokenizer BOS token')
        text = tokenizer.bos_token + prompt
        # Space-prefixed answer labels match the original completion protocol.
        labels = [' ' + letter for letter in CHOICES]
    else:
        subject = row['subject'].replace('_', ' ')
        messages = []
        for example in demonstrations:
            messages.extend([{'role': 'user', 'content': format_example(example)},
                             {'role': 'assistant', 'content': example['gold_answer']}])
        instruction = f'Answer this multiple-choice question about {subject}. Choose one letter among A, B, C, D.\n\n'
        messages.append({'role': 'user', 'content': instruction + format_example(row)})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        labels = list(CHOICES)
    enc = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    ids = enc['input_ids'][0].tolist()
    if len(ids) > args.max_input_length:
        raise ValueError(f'MMLU {row["id"]} ({row["subject"]}) has {len(ids)} tokens; increase --max_input_length. No examples were dropped.')
    label_ids = []
    for label in labels:
        full = tokenizer(text + label, add_special_tokens=False)['input_ids']
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise ValueError(f'Label {label!r} is not a single continuation token; this tokenizer needs a different scoring protocol')
        label_ids.append(full[-1])
    if len(set(label_ids)) != 4:
        raise ValueError('MMLU candidate labels must map to four distinct tokens')
    return enc, label_ids


def choose_answer(scores):
    if len(scores) != 4 or not all(math.isfinite(x) for x in scores):
        raise ValueError('Expected four finite MMLU label log-probabilities')
    return CHOICES[max(range(4), key=lambda i: scores[i])]


@torch.no_grad()
def score_one(args, tokenizer, model, row, demonstrations):
    enc, label_ids = tokenize_prompt(tokenizer, row, demonstrations, args)
    enc = {k: v.to(model.get_input_embeddings().weight.device) for k, v in enc.items()}
    extra = {'logits_to_keep': 1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    output = model(**enc, use_cache=False, **extra)
    logits = output.logits[0, -1].float()
    # Full-vocabulary log-probabilities; choice softmax below is conditional on A-D.
    scores = logits[label_ids] - torch.logsumexp(logits, dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError(f'Non-finite model scores at {row["id"]}')
    probabilities = torch.softmax(scores, dim=-1).tolist()
    scores = scores.tolist()
    return {'prediction_answer': choose_answer(scores), 'choice_logprobs': dict(zip(CHOICES, scores)),
            'choice_probabilities': dict(zip(CHOICES, probabilities)), 'n_prompt_tokens': enc['input_ids'].shape[-1]}


def aggregate(rows):
    n, correct = len(rows), sum(row['correct'] for row in rows)
    return {'n_questions': n, 'correct': correct, 'accuracy': correct/n if n else None}


def metrics_for(rows):
    result = aggregate(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['subject']].append(row)
    result['by_subject'] = {subject: aggregate(group) for subject, group in sorted(groups.items())}
    result['macro_subject_accuracy'] = sum(m['accuracy'] for m in result['by_subject'].values()) / len(groups)
    return result


def compare_to_baseline(metrics, baseline):
    if baseline is None:
        return
    def compare(current, reference):
        current['baseline_accuracy'] = reference['accuracy']
        current['accuracy_delta_vs_baseline'] = current['accuracy'] - reference['accuracy']
        current['accuracy_drop_vs_baseline'] = reference['accuracy'] - current['accuracy']
    compare(metrics, baseline)
    for subject, result in metrics['by_subject'].items():
        compare(result, baseline['by_subject'][subject])
    metrics['macro_subject_accuracy_delta_vs_baseline'] = metrics['macro_subject_accuracy'] - baseline['macro_subject_accuracy']


def paths(args, country=None, facet=None, features=None):
    condition = 'baseline' if country is None else f'sae_disabled_{sae_eval.base.slugify(country)}_{sae_eval.base.slugify(facet)}_{digest(features)[:10]}'
    stem = f'mmlu_test_{args.shots}shot_{args.run_id}_{condition}'
    return args.output_dir / f'predictions_{stem}.json', args.output_dir / f'metrics_{stem}.json'


def evaluate(args, tokenizer, model, data, demonstrations, country=None, facet=None, features=None, baseline=None):
    prediction_path, metric_path = paths(args, country, facet, features)
    metadata = {'run_id': args.run_id, 'evaluation_config': args.evaluation_config,
                'condition': 'baseline' if country is None else 'sae_disabled', 'country': country, 'facet': facet,
                'disabled_features_by_layer': {f'L{k}R': v for k,v in (features or {}).items()},
                'n_disabled_features': sum(map(len, (features or {}).values()))}
    rows = []
    if (args.resume_partial or args.skip_existing) and prediction_path.exists():
        checkpoint = json.loads(prediction_path.read_text())
        if checkpoint['metadata'] != metadata or len(checkpoint['rows']) > len(data):
            raise ValueError(f'Checkpoint configuration mismatch: {prediction_path}')
        rows = checkpoint['rows']
        for source, row in zip(data, rows):
            if any(row.get(k) != v for k,v in source.items()):
                raise ValueError(f'Checkpoint question mismatch: {prediction_path}')
            scores = [row['choice_logprobs'][letter] for letter in CHOICES]
            predicted = choose_answer(scores)
            probabilities = [row['choice_probabilities'][letter] for letter in CHOICES]
            expected = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=-1).tolist()
            if row['prediction_answer'] != predicted or row['correct'] != int(predicted == source['gold_answer']) or any(not math.isfinite(p) or not math.isclose(p,e,abs_tol=1e-6) for p,e in zip(probabilities,expected)):
                raise ValueError(f'Checkpoint score mismatch: {prediction_path}')
        if not args.resume_partial and not (args.skip_existing and len(rows) == len(data) and metric_path.exists()):
            rows = []
    if len(rows) < len(data):
        hooks = sae_eval.load_hooks(model, features, args.expansion, args.dtype) if features else []
        with ExitStack() as stack:
            for layer, hook in hooks:
                handle = model.model.layers[layer].register_forward_hook(hook)
                stack.callback(handle.remove)
            for i in tqdm(range(len(rows),len(data)), desc=f'{country}/{facet}' if country else 'baseline'):
                row = data[i]
                result = score_one(args, tokenizer, model, row, demonstrations.get(row['subject'], []))
                rows.append({**row, **result, 'correct': int(result['prediction_answer'] == row['gold_answer'])})
                if (i+1) % args.save_every == 0:
                    write_json(prediction_path, {'metadata': metadata, 'rows': rows})
    write_json(prediction_path, {'metadata': metadata, 'rows': rows})
    metrics = {**metadata, 'dataset': DATASET, 'split': 'test', 'setting': f'{args.shots}-shot',
               'metric': 'micro_accuracy', **metrics_for(rows), 'predictions_file': str(prediction_path)}
    compare_to_baseline(metrics, baseline)
    write_json(metric_path, metrics)
    print(f'[result] {country or "baseline"}/{facet or ""}: {metrics["accuracy"]:.4f} ({metrics["correct"]}/{metrics["n_questions"]})')
    return metrics


def main():
    args = parse_args()
    groups = sae_eval.load_groups(args)
    for (country, facet), features in groups.items():
        print(f'[plan] {country}/{facet}: {sum(map(len, features.values()))} features across {len(features)} layers')
    if args.list_conditions:
        return
    import pandas as pd
    from huggingface_hub import hf_hub_download
    def read_split(split):
        file = f'all/{split}-00000-of-00001.parquet'
        path = args.dataset_dir/file if args.dataset_dir else Path(hf_hub_download(DATASET,file,repo_type='dataset',revision=REVISION))
        return pd.read_parquet(path).to_dict('records')
    data, demonstrations = prepare_data(read_split('test'), read_split('dev') if args.shots else [], args)
    args.evaluation_config = {'dataset': DATASET, 'dataset_revision': REVISION if args.dataset_dir is None else None,
        'data_sha256': digest(data), 'demonstrations_sha256': digest(demonstrations), 'demonstrations': demonstrations,
        'model_id': args.model_id, 'shots': args.shots, 'prompt_format': args.prompt_format, 'prompt_version': PROMPT_VERSION,
        'scoring': 'next_token_label_loglikelihood_v1', 'aggregation': 'micro_accuracy', 'dtype': args.dtype,
        'load_in_8bit': args.load_in_8bit, 'trust_remote_code': args.trust_remote_code,
        'max_input_length': args.max_input_length, 'sample_seed': args.sample_seed if args.n_samples is not None else None,
        'expansion': args.expansion, 'layers': list(sae_eval.LAYERS), 'num_features_per_layer': args.num_features_per_layer}
    sae_eval.base = sae_eval.load_base()
    sae_eval.base.add_sae_paths(str(sae_eval.ROOT))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code, local_files_only=Path(args.model_id).exists())
    # Validate token boundaries and full prompt lengths before loading model weights.
    for row in tqdm(data, desc='validate prompts'):
        tokenize_prompt(tokenizer,row,demonstrations.get(row['subject'],[]),args)
    args.evaluation_config['chat_template_sha256'] = digest(tokenizer.chat_template) if args.prompt_format == 'chat' else None
    args.run_id = digest(args.evaluation_config)[:16]
    print(f'[data] {len(data)} questions / {len({r["subject"] for r in data})} subjects; {args.shots}-shot; {args.prompt_format}; label likelihood')
    print(f'[output] {args.output_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and args.dtype != 'float32':
        raise ValueError('CPU execution requires --dtype float32')
    tokenizer, model = sae_eval.base.load_model_and_tokenizer(args.model_id,args.dtype,args.trust_remote_code,args.load_in_8bit,device)
    summary_path = args.output_dir/f'summary_metrics_mmlu_{args.shots}shot_{args.run_id}.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    def record(result):
        key = (result['condition'],result['country'],result['facet'])
        summary[:] = [m for m in summary if (m['condition'],m['country'],m['facet']) != key]
        summary.append(result)
        write_json(summary_path,summary)
    baseline = None
    if not args.skip_baseline:
        baseline = evaluate(args,tokenizer,model,data,demonstrations)
        record(baseline)
    else:
        baseline = next((m for m in summary if m['condition']=='baseline' and m['evaluation_config']==args.evaluation_config),None)
    for (country,facet),features in groups.items():
        record(evaluate(args,tokenizer,model,data,demonstrations,country,facet,features,baseline))
    print(f'[done] {summary_path}')


if __name__ == '__main__':
    main()
