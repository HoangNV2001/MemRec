#!/usr/bin/env python3
"""30-user, no-network wiring smoke for the full MemRec Books pipeline.

This validates data/candidate/Stage-R/Stage-ReRank/Stage-W plumbing only.
Fake scores are NOT research results and do not replace the H100 LLM smoke.
"""

import argparse
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.train.trainer_memrec as trainer_module
from src.data import RecDataset
from src.utils import load_config


class FakeJSONClient:
    """Deterministic schema-shaped responses; never opens an API connection."""

    def __init__(self, **kwargs):
        self.model = 'cpu-fake'
        self.api_endpoint = 'no-network'
        self.requests = 0
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_physical_requests = 0
        self.total_cache_hits = 0

    def generate_json(self, messages, properties, **kwargs):
        if getattr(self, 'request_budget', None) is not None:
            self.request_budget.consume()
        self.requests += 1
        self.total_requests += 1
        self.total_physical_requests += 1
        if 'scores' in properties:
            section = messages[0]['content'].split('**Candidate Item Memories:**', 1)[1]
            item_ids = [int(value) for value in re.findall(r'• Item (\d+) \(', section)]
            if len(item_ids) != 10 or len(set(item_ids)) != 10:
                raise ValueError('CPU smoke could not identify exactly 10 candidates')
            return {'scores': [
                {'item_id': item_id, 'score': 1.0 - idx / 20, 'rationale': 'CPU wiring smoke'}
                for idx, item_id in enumerate(item_ids)
            ]}
        if 'facets' in properties:
            return {'facets': [
                {'facet': 'CPU wiring smoke preference', 'confidence': 0.8, 'supporting_neighbors': []}
            ], 'support_edges': []}
        if 'user_memory' in properties:
            return {
                'user_memory': 'CPU wiring smoke user memory',
                'item_memory': 'CPU wiring smoke item memory',
                'neighbor_updates': [],
            }
        raise ValueError(f'Unknown JSON schema: {sorted(properties)}')

    def get_token_stats(self):
        return {
            'total_requests': self.total_requests,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': 0,
            'avg_input_tokens': 0,
            'avg_output_tokens': 0,
        }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--users', type=int, default=30)
    parser.add_argument('--warmup-user-scope', choices=['eval', 'all'], default='eval')
    parser.add_argument('--journal', action='store_true',
                        help='Exercise durable full-dev warm-up/eval resume with fake LLM')
    args = parser.parse_args()
    if not 20 <= args.users <= 30:
        parser.error('CPU smoke is intentionally limited to 20–30 users')

    config = load_config(str(ROOT / 'configs/memrec_instructrec-books_full_benchmark.yaml'))
    config.update({
        'n_eval_users': args.users,
        'eval_cohort': 'dev',
        'warmup_user_scope': args.warmup_user_scope,
        'eval_feedback': 'none',
        'llm_model': 'cpu-fake',
        'provider': {
            'name': 'openai', 'model': 'cpu-fake', 'revision': 'cpu-fake-v1',
            'endpoint': 'no-network', 'api_key': 'fake', 'sdk_max_retries': 0,
        },
    })
    output = ROOT / f'results/full_memrec_cpu_smoke_{args.users}_{args.warmup_user_scope}-hnv'
    if args.journal:
        if args.warmup_user_scope != 'all':
            parser.error('--journal requires --warmup-user-scope all')
        output = ROOT / f'results/full_memrec_cpu_smoke_{args.users}_all_journal-hnv'
        output.mkdir(parents=True, exist_ok=True)
        os.environ['MEMREC_BOOKS_RUN_JOURNAL_DB'] = str(output / 'journal.sqlite')
        os.environ['MEMREC_BOOKS_REQUEST_BUDGET_DB'] = str(output / 'request-budget.sqlite')
        os.environ['MEMREC_BOOKS_FULL_RUN_ID'] = output.name
        os.environ['MEMREC_BOOKS_FULL_GIT_COMMIT'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip()
        config['output_dir'] = str(output)
    trainer_module.LLMClient = FakeJSONClient
    dataset = RecDataset(
        str(ROOT / 'data/processed/instructrec-books/instructrec-books.inter'),
        seed=42,
        precompute_negatives=False,
    )
    trainer = trainer_module.MemRecTrainer(None, dataset, config, torch.device('cpu'))
    metrics = trainer.evaluate(split='test', save_dir=str(output), parallel=False)
    assert metrics['n_eval_users'] == args.users
    assert metrics['n_rankings'] == args.users
    assert metrics['n_failed_rankings'] == 0
    assert metrics['n_stage_r_calls'] == args.users
    assert metrics['n_stage_rr_calls'] == args.users
    expected_warmup = len(dataset.test_data) if args.warmup_user_scope == 'all' else args.users
    assert metrics['n_warmup_users'] == expected_warmup
    assert metrics['n_stage_w_warmup_calls'] == expected_warmup
    assert trainer.agent.n_stage_w_calls == expected_warmup
    print(
        f'CPU WIRING SMOKE PASS: {args.users} ranked / {expected_warmup} warmed; '
        'results are not model performance'
    )
    print(
        f'CPU RESOURCE: elapsed_seconds={time.perf_counter() - started:.2f} '
        f'peak_rss_mib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}'
    )


if __name__ == '__main__':
    main()
