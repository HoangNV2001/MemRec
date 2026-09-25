#!/usr/bin/env python3
"""Fail closed on denominator, journal and provenance of Books full-dev run."""

import argparse
import json
import math
import sqlite3
from pathlib import Path

from src.data.books_protocol import books_cohorts


def check(run_dir: Path) -> dict:
    results = list(run_dir.glob('instructrec-books_memrec_agent_seed42_*.json'))
    if not results:
        raise ValueError('No final metrics file')
    reports = [json.loads(path.read_text()) for path in sorted(results)]
    report = reports[-1]
    if any(old['config'] != report['config'] or
           old['test_metrics'] != report['test_metrics'] for old in reports[:-1]):
        raise ValueError('Resumed metrics files disagree')
    config, metrics = report['config'], report['test_metrics']
    if (config['eval_cohort'] != 'dev' or config['warmup_user_scope'] != 'all'
            or config['eval_feedback'] != 'none'
            or not config['use_pregenerated_candidates']
            or config['memrec']['reranker_mode'] != 'llm'):
        raise ValueError('Not the locked Books full-MemRec dev protocol')
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    if config['provider']['revision'] != manifest['model_revision']:
        raise ValueError('Model revision differs from full-run manifest')
    if (metrics['n_eval_users'] != 2000 or metrics['n_rankings'] != 2000
            or metrics['n_warmup_users'] != 7377
            or metrics['n_stage_r_calls'] != 2000
            or metrics['n_stage_rr_calls'] != 2000
            or metrics['n_stage_w_calls'] != 0):
        raise ValueError('Wrong evaluation/warm-up denominator or stages')
    if (metrics['n_stage_r_warmup_calls'] != 7377
            or metrics['n_stage_rr_warmup_calls'] != 7377):
        raise ValueError('Full warm-up omitted Stage-R or ReRank for users')
    if metrics['llm_physical_requests'] > metrics['llm_request_hard_cap']:
        raise ValueError('LLM hard cap exceeded')

    cohorts, _ = books_cohorts(list(range(7377)))
    predictions = [json.loads(line) for line in
                   (run_dir / 'test_predictions.jsonl').read_text().splitlines()]
    if [row['user_id'] for row in predictions] != cohorts['dev']:
        raise ValueError('Dev predictions missing, duplicated or out of order')
    failures = 0
    for row in predictions:
        candidates, ranked = row['candidates'], row['ranked_items']
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError(f"Invalid candidates for user {row['user_id']}")
        if row['failure'] is None:
            if len(ranked) != 10 or set(ranked) != set(candidates):
                raise ValueError(f"Malformed successful ranking for user {row['user_id']}")
            if ranked[row['target_position']] != row['target_item']:
                raise ValueError(f"Wrong position for user {row['user_id']}")
        else:
            failures += 1
    if failures != metrics['n_failed_rankings']:
        raise ValueError('Failure count differs between metrics and predictions')
    if (run_dir / 'heldout_predictions.jsonl').exists():
        raise ValueError('Held-out predictions are not allowed')

    with sqlite3.connect(run_dir / 'journal.sqlite') as connection:
        counts = dict(connection.execute(
            'SELECT phase, COUNT(*) FROM entries GROUP BY phase'
        ).fetchall())
    if counts != {'warmup': 7377, 'eval': 2000}:
        raise ValueError(f'Incomplete full-dev progress journal: {counts}')
    with sqlite3.connect(run_dir / 'request-budget.sqlite') as connection:
        limit, used = connection.execute(
            'SELECT lim, used FROM budget WHERE id=1'
        ).fetchone()
    if (limit != metrics['llm_request_hard_cap']
            or used != metrics['llm_physical_requests']):
        raise ValueError('Durable request budget differs from final metrics')
    for metric in ('NDCG@5', 'Hit@1'):
        if not math.isfinite(metrics[metric]):
            raise ValueError(f'Non-finite {metric}')
    return {'status': 'pass', 'n_dev_users': 2000, 'n_warmup_users': 7377,
            'n_failed_rankings': failures, 'physical_attempts_reserved': used,
            'ndcg_at_5': metrics['NDCG@5'], 'hit_at_1': metrics['Hit@1']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.run_dir), sort_keys=True))


if __name__ == '__main__':
    main()
