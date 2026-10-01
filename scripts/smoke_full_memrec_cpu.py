#!/usr/bin/env python3
"""30-user, no-network wiring smoke for the full MemRec Books pipeline.

This validates data/candidate/Stage-R/Stage-ReRank/Stage-W plumbing only.
Fake scores are NOT research results and do not replace the H100 LLM smoke.
"""

import argparse
import json
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
                        help='Exercise durable Books warm-up/eval resume with fake LLM')
    parser.add_argument('--dev700', action='store_true',
                        help='CPU-only 700 warm-up / 200 evaluation wiring test')
    parser.add_argument('--graph-walk3', action='store_true',
                        help='Use the frozen graph-walk evidence selector')
    parser.add_argument('--transition-stage-r', action='store_true',
                        help='Use train-only directed one-step evidence in Stage-R')
    parser.add_argument('--cmirank-fidelity-trace', action='store_true',
                        help='Aggregate target-blind pruner/packer behavior on 20–30 users')
    args = parser.parse_args()
    if args.graph_walk3 and args.transition_stage_r:
        parser.error('Choose only one evidence selector')
    if args.dev700:
        if args.users != 30 or args.warmup_user_scope != 'eval':
            parser.error('--dev700 uses its fixed 700/200 cohort, not --users or --warmup-user-scope')
        args.users = 200
        args.warmup_user_scope = 'subset'
    elif not 20 <= args.users <= 30:
        parser.error('CPU smoke is intentionally limited to 20–30 users')
    if args.cmirank_fidelity_trace and (args.dev700 or args.warmup_user_scope != 'eval'):
        parser.error('CM-IRank fidelity trace requires a 20–30-user eval-scope smoke')

    config_name = ('memrec_instructrec-books_dev700_transition_stage_r.yaml'
                   if args.transition_stage_r else
                   'memrec_instructrec-books_dev700_graph_walk3.yaml'
                   if args.graph_walk3 else
                   'memrec_instructrec-books_dev700.yaml' if args.dev700
                   else 'memrec_instructrec-books_full_benchmark.yaml')
    config = load_config(str(ROOT / 'configs' / config_name))
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
    variant = '_transition_stage_r' if args.transition_stage_r else '_graph_walk3' if args.graph_walk3 else ''
    output = ROOT / f'results/full_memrec_cpu_smoke_{args.users}_{args.warmup_user_scope}{variant}-hnv'
    if args.journal:
        output = ROOT / f'results/full_memrec_cpu_smoke_{args.users}_{args.warmup_user_scope}{variant}_journal-hnv'
        if args.warmup_user_scope == 'eval':
            config['books_subset_users'] = 700
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
    prune_rows = []
    pack_rows = []
    if args.cmirank_fidelity_trace:
        original_prune = trainer.agent.pruner.prune
        original_pack = trainer.agent.packer.pack

        def traced_prune(*prune_args, **prune_kwargs):
            result = original_prune(*prune_args, **prune_kwargs)
            neighbors = result.get('neighbors', [])
            items = [row for row in neighbors if row['type'] == 'item']
            prune_rows.append({
                'selected': len(neighbors),
                'items': len(items),
                'users': len(neighbors) - len(items),
                'item_metadata_overlap_placeholder': sum(
                    row.get('metadata_overlap') == 0.5 for row in items),
                'zero_memory_similarity': sum(
                    row.get('memory_sim') == 0.0 for row in neighbors),
            })
            return result

        def traced_pack(*pack_args, **pack_kwargs):
            result = original_pack(*pack_args, **pack_kwargs)
            pack_rows.append({
                'packed': result['n_neighbors'],
                'estimated_tokens': result['estimated_tokens'],
                'has_user_memory_summary': bool(pack_kwargs.get('user_memory_summary')),
            })
            return result

        trainer.agent.pruner.prune = traced_prune
        trainer.agent.packer.pack = traced_pack
    metrics = trainer.evaluate(split='test', save_dir=str(output), parallel=False)
    assert metrics['n_eval_users'] == args.users
    assert metrics['n_rankings'] == args.users
    assert metrics['n_failed_rankings'] == 0
    assert metrics['n_stage_r_calls'] == args.users
    assert metrics['n_stage_rr_calls'] == args.users
    expected_warmup = (len(dataset.test_data) if args.warmup_user_scope == 'all'
                       else 700 if args.warmup_user_scope == 'subset' else args.users)
    assert metrics['n_warmup_users'] == expected_warmup
    assert metrics['n_stage_w_warmup_calls'] == expected_warmup
    assert trainer.agent.n_stage_w_calls == expected_warmup
    if args.cmirank_fidelity_trace:
        if len(prune_rows) != 2 * args.users or len(pack_rows) != len(prune_rows):
            raise AssertionError('Unexpected CM-IRank trace call count')
        def summarize(start, end):
            selected = prune_rows[start:end]
            packed = pack_rows[start:end]
            return {
                'calls': len(selected),
                'selected_neighbors_total': sum(row['selected'] for row in selected),
                'packed_neighbors_total': sum(row['packed'] for row in packed),
                'item_neighbors_total': sum(row['items'] for row in selected),
                'item_metadata_overlap_placeholder_total': sum(
                    row['item_metadata_overlap_placeholder'] for row in selected),
                'zero_memory_similarity_total': sum(
                    row['zero_memory_similarity'] for row in selected),
                'has_user_memory_summary_calls': sum(
                    row['has_user_memory_summary'] for row in packed),
                'estimated_tokens_max': max(row['estimated_tokens'] for row in packed),
            }
        report = {
            'status': 'CPU_FAKE_LLM_FIDELITY_TRACE_NOT_RANKING_RESULT',
            'warmup': summarize(0, args.users),
            'evaluation': summarize(args.users, 2 * args.users),
            'test_label_stage_w_calls': 0,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / 'cmirank_fidelity_trace.json').write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print('CM-IRANK G0 TRACE: ' + json.dumps(report, sort_keys=True))
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
