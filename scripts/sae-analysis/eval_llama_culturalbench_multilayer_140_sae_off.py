#!/usr/bin/env python
"""Zero-shot CulturalBench-Hard baseline and seven-layer SAE-off evaluation.

Each country/facet intervention runs on the same full dataset. --country and
--facet filter interventions, not questions. --n_samples counts complete groups
of four options. Results, checkpoints and metrics are JSON only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re

import torch
from tqdm import tqdm
import eval_llama_gsm8k_multilayer_140_sae_off as sae_eval

DATASET = 'kellycyy/CulturalBench'
REVISION = '996d62a55f6a49c13596f0a31b772812a1aacd67'
PROMPT = ('Question: {question}\n\nAnswer: {option}\n\n'
          'Is this answer true or false for this question? You must choose either True or False.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


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
    parser.add_argument('--max_input_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=2, help='Published protocol: at most 2 output tokens')
    parser.add_argument('--answer_parser', choices=('strict', 'leading'), default='strict',
                        help='strict: label only (published criterion); leading: accept an initial label plus explanation (custom protocol)')
    parser.add_argument('--n_samples', type=int, help='First N complete questions (4N generations)')
    parser.add_argument('--output_dir', type=Path, help='Default: DATA_DIR/culturalbench')
    parser.add_argument('--save_every', type=int, default=20, help='Checkpoint every N binary responses')
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--resume_partial', action='store_true')
    parser.add_argument('--list_conditions', action='store_true')
    parser.add_argument('--dataset_file', type=Path, help='Optional local CulturalBench-Hard CSV')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = sae_eval.default_output_dir().with_name('culturalbench')
    for field in ('num_features_per_layer', 'max_input_length', 'max_new_tokens', 'save_every', 'n_samples'):
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f'--{field} must be positive')
    return args


def prepare_data(records, n_samples=None):
    """Validate the published schema before selecting complete question groups."""
    groups = {}
    seen = set()
    for row in records:
        idx, qid = int(row['data_idx']), int(row['question_idx'])
        if idx in seen:
            raise ValueError(f'Duplicate data_idx: {idx}')
        seen.add(idx)
        label = str(row['answer']).strip().lower()
        if label not in ('true', 'false'):
            raise ValueError(f'Invalid True/False label for data_idx {idx}')
        normalized = {'data_idx': idx, 'question_idx': qid,
                      'question': str(row['prompt_question']), 'option': str(row['prompt_option']),
                      'gold_answer': label == 'true', 'question_country': str(row['country']).strip()}
        groups.setdefault(qid, []).append(normalized)
    for qid, options in groups.items():
        if len(options) != 4 or len({(r['question'], r['question_country']) for r in options}) != 1:
            raise ValueError(f'Question {qid} must have four options with consistent question and country')
    selected = list(groups.values())[:n_samples]
    result = [row for options in selected for row in options]
    if not result:
        raise ValueError('No CulturalBench questions found')
    return result


def parse_answer(text, mode='strict'):
    """Published criterion by default; optional prefix extraction is explicit.

    Never infer a label from a word occurring inside an explanation. Reject
    immediate alternatives/negations in leading mode as ambiguous responses.
    """
    if mode == 'strict':
        match = re.fullmatch(r'\s*(true|false)[.!]?\s*', text, flags=re.IGNORECASE)
    elif mode == 'leading':
        match = re.match(r'\s*(true|false)\b', text, flags=re.IGNORECASE)
        if match and re.match(r'\s*(?:[/,]|or\b|and\b|nor\b|not\b)', text[match.end():], flags=re.IGNORECASE):
            return None
    else:
        raise ValueError(f'Unknown answer parser: {mode}')
    return None if match is None else match.group(1).lower() == 'true'


@torch.no_grad()
def generate_one(args, tokenizer, model, row):
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': PROMPT.format(question=row['question'], option=row['option'])}],
        tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    length = enc['input_ids'].shape[-1]
    if length > args.max_input_length:
        raise ValueError(f'Prompt exceeds --max_input_length for data_idx {row["data_idx"]}: {length}')
    enc = {k: v.to(model.get_input_embeddings().weight.device) for k, v in enc.items()}
    stops = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids('<|eot_id|>')
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        stops.append(eot)
    output = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                            eos_token_id=list(set(stops)), pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][length:], skip_special_tokens=True).strip()


def aggregate(rows):
    questions = defaultdict(list)
    for row in rows:
        questions[row['question_idx']].append(row)
    correct = sum(len(options) == 4 and all(r['correct'] for r in options) for options in questions.values())
    return {'n_questions': len(questions), 'correct': correct,
            'accuracy': correct / len(questions) if questions else None,
            'n_binary_responses': len(rows),
            'binary_accuracy': sum(r['correct'] for r in rows) / len(rows) if rows else None,
            'invalid_responses': sum(r['prediction_answer'] is None for r in rows)}


def metrics_for(rows, country=None):
    result = aggregate(rows)
    countries = sorted({r['question_country'] for r in rows})
    result['by_question_country'] = {c: aggregate([r for r in rows if r['question_country'] == c]) for c in countries}
    if country is not None:
        target = country.casefold()
        result['target_country'] = aggregate([r for r in rows if r['question_country'].casefold() == target])
        result['other_countries'] = aggregate([r for r in rows if r['question_country'].casefold() != target])
    return result


def compare_to_baseline(metrics, baseline):
    if baseline is None:
        return metrics
    def add_delta(current, reference):
        if current['accuracy'] is not None and reference['accuracy'] is not None:
            current['baseline_accuracy'] = reference['accuracy']
            current['accuracy_delta_vs_baseline'] = current['accuracy'] - reference['accuracy']
            current['accuracy_drop_vs_baseline'] = reference['accuracy'] - current['accuracy']
    add_delta(metrics, baseline)
    for country, values in metrics['by_question_country'].items():
        add_delta(values, baseline['by_question_country'][country])
    if metrics.get('country') is not None:
        target = metrics['country'].casefold()
        target_baseline = [v for c, v in baseline['by_question_country'].items() if c.casefold() == target]
        if target_baseline:
            add_delta(metrics['target_country'], target_baseline[0])
        # Compute the reference from question counts rather than a mean of countries.
        other = [v for c, v in baseline['by_question_country'].items() if c.casefold() != target]
        n = sum(v['n_questions'] for v in other)
        add_delta(metrics['other_countries'], {'accuracy': sum(v['correct'] for v in other) / n if n else None})
    return metrics


def paths(args, country=None, facet=None, features=None):
    condition = 'baseline' if country is None else f'sae_disabled_{sae_eval.base.slugify(country)}_{sae_eval.base.slugify(facet)}_{digest(features)[:10]}'
    stem = f'culturalbench_hard_{args.run_id}_{condition}'
    return args.output_dir / f'predictions_{stem}.json', args.output_dir / f'metrics_{stem}.json'


def evaluate(args, tokenizer, model, data, country=None, facet=None, features=None, baseline=None):
    prediction_path, metric_path = paths(args, country, facet, features)
    metadata = {'run_id': args.run_id, 'evaluation_config': args.evaluation_config,
                'condition': 'baseline' if country is None else 'sae_disabled', 'country': country, 'facet': facet,
                'disabled_features_by_layer': {f'L{k}R': v for k, v in (features or {}).items()},
                'n_disabled_features': sum(map(len, (features or {}).values()))}
    rows = []
    if (args.resume_partial or args.skip_existing) and prediction_path.exists():
        checkpoint = json.loads(prediction_path.read_text())
        if checkpoint['metadata'] != metadata or len(checkpoint['rows']) > len(data):
            raise ValueError(f'Checkpoint configuration mismatch: {prediction_path}')
        rows = checkpoint['rows']
        for source, row in zip(data, rows):
            if any(row.get(key) != value for key, value in source.items()) or 'prediction_text' not in row:
                raise ValueError(f'Checkpoint question mismatch: {prediction_path}')
            parsed = parse_answer(row['prediction_text'], args.answer_parser)
            if row.get('prediction_answer') is not parsed or row.get('correct') != int(parsed is not None and parsed == source['gold_answer']):
                raise ValueError(f'Checkpoint answer mismatch: {prediction_path}')
        if not args.resume_partial and not (args.skip_existing and len(rows) == len(data) and metric_path.exists()):
            rows = []
    if len(rows) < len(data):
        hooks = sae_eval.load_hooks(model, features, args.expansion, args.dtype) if features else []
        with ExitStack() as stack:
            for layer, hook in hooks:
                handle = model.model.layers[layer].register_forward_hook(hook)
                stack.callback(handle.remove)
            for i in tqdm(range(len(rows), len(data)), desc=f'{country}/{facet}' if country else 'baseline'):
                text = generate_one(args, tokenizer, model, data[i])
                prediction = parse_answer(text, args.answer_parser)
                rows.append({**data[i], 'prediction_text': text, 'prediction_answer': prediction,
                             'correct': int(prediction is not None and prediction == data[i]['gold_answer'])})
                if (i + 1) % args.save_every == 0:
                    write_json(prediction_path, {'metadata': metadata, 'rows': rows})
    write_json(prediction_path, {'metadata': metadata, 'rows': rows})
    metrics = {**metadata, 'dataset': DATASET, 'setting': 'zero-shot', 'metric': 'all_four_options_correct',
               **metrics_for(rows, country), 'predictions_file': str(prediction_path)}
    compare_to_baseline(metrics, baseline)
    write_json(metric_path, metrics)
    print(f'[result] {country or "baseline"}/{facet or ""}: {metrics["accuracy"]:.4f} ({metrics["correct"]}/{metrics["n_questions"]})')
    if metrics['invalid_responses']:
        print(f'[format] {metrics["invalid_responses"]}/{len(rows)} responses did not match the {args.answer_parser} parser')
    return metrics


def main():
    args = parse_args()
    groups = sae_eval.load_groups(args)
    for (country, facet), features in groups.items():
        print(f'[plan] {country}/{facet}: {sum(map(len, features.values()))} features across {len(features)} layers; all questions')
    if args.list_conditions:
        return
    from huggingface_hub import hf_hub_download
    import pandas as pd
    file = args.dataset_file or Path(hf_hub_download(DATASET, 'CulturalBench-Hard.csv', repo_type='dataset', revision=REVISION))
    data = prepare_data(pd.read_csv(file).to_dict('records'), args.n_samples)
    args.evaluation_config = {'model_id': args.model_id, 'dataset': DATASET, 'dataset_revision': REVISION if args.dataset_file is None else None,
        'data_sha256': digest(data), 'prompt': PROMPT, 'prompt_protocol': 'paper_appendix_d_user_only_v1',
        'parser': f'{args.answer_parser}_boolean_v2', 'metric': 'all_four_options_correct', 'dtype': args.dtype,
        'load_in_8bit': args.load_in_8bit, 'trust_remote_code': args.trust_remote_code,
        'max_input_length': args.max_input_length, 'max_new_tokens': args.max_new_tokens,
        'do_sample': False, 'expansion': args.expansion, 'layers': list(sae_eval.LAYERS), 'num_features_per_layer': args.num_features_per_layer}
    args.run_id = digest(args.evaluation_config)[:16]
    print(f'[data] {len(data)//4} complete questions, {len(data)} binary responses per condition')
    print(f'[output] {args.output_dir}')
    print(f'[protocol] zero-shot, max_new_tokens={args.max_new_tokens}, parser={args.answer_parser}')
    if args.max_new_tokens != 2 or args.answer_parser != 'strict':
        print('[protocol] Custom output settings differ from the published 2-token, label-only criterion')
    sae_eval.base = sae_eval.load_base()
    sae_eval.base.add_sae_paths(str(sae_eval.ROOT))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and args.dtype != 'float32':
        raise ValueError('CPU execution requires --dtype float32')
    tokenizer, model = sae_eval.base.load_model_and_tokenizer(args.model_id, args.dtype, args.trust_remote_code, args.load_in_8bit, device)
    summary_path = args.output_dir / f'summary_metrics_culturalbench_hard_{args.run_id}.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    def record(result):
        key = (result['condition'], result['country'], result['facet'])
        summary[:] = [m for m in summary if (m['condition'], m['country'], m['facet']) != key]
        summary.append(result)
        write_json(summary_path, summary)
    baseline = None
    if not args.skip_baseline:
        baseline = evaluate(args, tokenizer, model, data)
        record(baseline)
    else:
        baseline = next((m for m in summary if m['condition'] == 'baseline' and m['evaluation_config'] == args.evaluation_config), None)
    for (country, facet), features in groups.items():
        record(evaluate(args, tokenizer, model, data, country, facet, features, baseline))
    print(f'[done] {summary_path}')


if __name__ == '__main__':
    main()
