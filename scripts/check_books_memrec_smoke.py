#!/usr/bin/env python3
"""Fail closed on the fixed-list, 30-user, real-LLM Books smoke artifacts."""

import argparse
import hashlib
import json
from pathlib import Path


def check(run_dir: Path, expected_users: int = 30, allow_cache: bool = False,
          reference_predictions: Path | None = None) -> dict:
    result_files = list(run_dir.glob('instructrec-books_memrec_agent_seed42_*.json'))
    if len(result_files) != 1:
        raise ValueError(f'Expected exactly one metrics file, found {len(result_files)}')
    payload = json.loads(result_files[0].read_text())
    config = payload['config']
    metrics = payload['test_metrics']
    if config['eval_cohort'] != 'dev' or config['warmup_user_scope'] != 'eval':
        raise ValueError('Smoke must use 30 dev users and eval-only warm-up')
    if config['eval_feedback'] != 'none' or not config['use_pregenerated_candidates']:
        raise ValueError('Test-label feedback or non-original candidates detected')
    if config['memrec']['reranker_mode'] != 'llm':
        raise ValueError('Full MemRec LLM reranker was not used')
    if not config['provider'].get('revision'):
        raise ValueError('Unpinned LLM checkpoint')
    if (metrics['n_eval_users'] != expected_users
            or metrics['n_rankings'] != expected_users
            or metrics['n_warmup_users'] != expected_users):
        raise ValueError('Wrong smoke sample denominator')
    if metrics['n_failed_rankings'] != 0:
        raise ValueError('At least one ranking was malformed or failed')
    if metrics['n_stage_r_warmup_calls'] != expected_users:
        raise ValueError('Stage-R warm-up count mismatch')
    if metrics['n_stage_rr_warmup_calls'] != expected_users:
        raise ValueError('Stage-ReRank warm-up count mismatch')
    if metrics['n_stage_w_warmup_calls'] != expected_users:
        raise ValueError('Stage-W warm-up count mismatch')
    if metrics['n_stage_r_calls'] != expected_users or metrics['n_stage_rr_calls'] != expected_users:
        raise ValueError('Evaluation Stage-R or Stage-ReRank was skipped')
    if metrics['n_stage_w_calls'] != 0:
        raise ValueError('Test feedback unexpectedly invoked Stage-W')
    requests = metrics['llm_physical_requests']
    if (requests < 0 or requests > expected_users * 5
            or requests > metrics['llm_request_hard_cap']
            or (not allow_cache and requests != expected_users * 5)):
        raise ValueError('Unexpected physical LLM request count')
    if allow_cache and (
            config.get('books_subset_users') != 700 or reference_predictions is None):
        raise ValueError('Cache-backed smoke needs the locked subset and reference predictions')
    predictions_file = run_dir / 'test_predictions.jsonl'
    predictions = [json.loads(line) for line in predictions_file.read_text().splitlines()]
    if len(predictions) != expected_users:
        raise ValueError('Prediction count mismatch')
    if len({row['user_id'] for row in predictions}) != expected_users:
        raise ValueError('Duplicate smoke user')
    for row in predictions:
        candidates = row['candidates']
        ranked = row['ranked_items']
        if row['failure'] is not None or len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError(f"Invalid candidate or failure for user {row['user_id']}")
        if len(ranked) != 10 or set(ranked) != set(candidates):
            raise ValueError(f"Incomplete permutation for user {row['user_id']}")
        if ranked[row['target_position']] != row['target_item']:
            raise ValueError(f"Target position mismatch for user {row['user_id']}")
    if (run_dir / 'heldout_predictions.jsonl').exists():
        raise ValueError('Held-out predictions must remain sealed')
    if allow_cache:
        own_sha = hashlib.sha256(predictions_file.read_bytes()).hexdigest()
        reference_sha = hashlib.sha256(reference_predictions.read_bytes()).hexdigest()
        if own_sha != reference_sha:
            raise ValueError('Cache-backed subset smoke differs from promoted 30-user smoke')
    return {'status': 'pass', 'n_users': expected_users, 'physical_requests': requests,
            'model_revision': config['provider']['revision']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--allow-cache', action='store_true')
    parser.add_argument('--reference-predictions', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.run_dir, allow_cache=args.allow_cache,
                           reference_predictions=args.reference_predictions), sort_keys=True))


if __name__ == '__main__':
    main()
