#!/usr/bin/env python3
"""Frozen temporal-transition transfer on the 200-user full-MemRec Books run.

This is a post-ranking augmentation of *completed full MemRec*, not a change
to Stage-R or Stage-W. It uses no new LLM calls or GPU. Smoke must complete
before full evaluation; neither phase selects hyperparameters from labels.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_books_memrec_full_dev import check as check_baseline
from src.data import RecDataset
from src.temporal_books.common import sha256_file, stable_order
from src.temporal_common.graph import (
    build_transition_graph_from_users,
    one_step_scores,
    ppr_monte_carlo_scores,
    recent_seed_items,
)
from src.temporal_common.metrics import paired_bootstrap_ci, residual_ranking


BASELINE = ROOT / 'results/full_memrec_books_baselines/books-memrec-llm-dev700-v1-hnv'
OUTPUT = ROOT / 'results/full_memrec_books_baselines/books-memrec-transition-transfer-v1-hnv'
DATA = ROOT / 'data/processed/instructrec-books/instructrec-books.inter'
SMOKE_USERS = 30
FULL_USERS = 200
SEED_ITEMS = 6
RESTART_PROBABILITY = 0.15
MONTE_CARLO_WALKS = 50_000
MAX_WALK_STEPS = 64
ALPHA = 0.80
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260928


def load_baseline(run_dir: Path):
    completion = json.loads((run_dir / 'completion.json').read_text())
    if completion['status'] != 'passed':
        raise ValueError('Baseline completion is not passed')
    for name, expected in completion['artifact_sha256'].items():
        if sha256_file(run_dir / name) != expected:
            raise ValueError(f'Baseline artifact hash mismatch: {name}')
    gate = check_baseline(run_dir, FULL_USERS, 700, 'subset')
    if gate['status'] != 'pass':
        raise ValueError('Baseline 700/200 gate failed')
    rows = [json.loads(line) for line in (run_dir / 'test_predictions.jsonl').read_text().splitlines()]
    if len(rows) != FULL_USERS:
        raise ValueError('Wrong baseline denominator')
    return rows


def build_graph(dataset: RecDataset):
    # .inter timestamp is only a per-user sequence position. Train-only
    # histories, not validation/test targets, supply the directed edges.
    histories = (
        [(position + 1, str(item_id)) for position, item_id in enumerate(items)]
        for items in dataset.train_data.values()
    )
    graph, stats = build_transition_graph_from_users(histories, cutoff=10**9)
    expected_pairs = sum(max(0, len(items) - 1) for items in dataset.train_data.values())
    if stats['temporal_batch_pairs'] != expected_pairs:
        raise ValueError('Directed graph contains unexpected transition pairs')
    return graph, stats


def score_row(row: dict, dataset: RecDataset, graph: dict) -> dict:
    user_id = int(row['user_id'])
    candidates = [str(item) for item in row['candidates']]
    local = [str(item) for item in row['ranked_items']]
    if len(candidates) != 10 or len(set(candidates)) != 10:
        raise ValueError(f'Invalid original candidates for user {user_id}')
    if len(local) != 10 or set(local) != set(candidates):
        raise ValueError(f'Invalid baseline ranking for user {user_id}')
    pretest = [*dataset.train_data[user_id], dataset.valid_data[user_id]]
    if set(row['candidates']) & set(pretest):
        raise ValueError(f'Candidate overlaps pre-test history for user {user_id}')
    events = [(position + 1, str(item)) for position, item in enumerate(pretest)]
    seeds = recent_seed_items(events, len(events) + 1, limit=SEED_ITEMS)
    one_step = one_step_scores(seeds, candidates, graph)
    ppr = ppr_monte_carlo_scores(
        seeds, candidates, graph, walks=MONTE_CARLO_WALKS,
        restart_probability=RESTART_PROBABILITY, max_steps=MAX_WALK_STEPS,
        seed=stable_order(f'books-memrec-transition-v1\0{user_id}'),
    )
    if any(not math.isfinite(value) or value < 0 for value in [*one_step.values(), *ppr.values()]):
        raise ValueError(f'Invalid transition score for user {user_id}')
    # Malformed baseline outputs remain misses; graph must not silently repair
    # LLM syntax failures and inflate the paired method score.
    valid = row['failure'] is None
    one_rank = residual_ranking(candidates, local, one_step, alpha=ALPHA) if valid else local
    ppr_rank = residual_ranking(candidates, local, ppr, alpha=ALPHA) if valid else local
    return {
        'user_id': user_id, 'candidates': row['candidates'],
        'local_ranking': row['ranked_items'], 'failure': row['failure'],
        'seed_item_ids': [int(item) for item in seeds],
        'one_step_scores': one_step, 'ppr_scores': ppr,
        'one_step_ranking': [int(item) for item in one_rank],
        'ppr_ranking': [int(item) for item in ppr_rank],
    }


def validate_score_row(scored: dict, source: dict):
    if (scored['user_id'] != source['user_id']
            or scored['candidates'] != source['candidates']
            or scored['local_ranking'] != source['ranked_items']
            or scored['failure'] != source['failure']):
        raise ValueError('Transition output differs from frozen MemRec input')
    candidates = set(source['candidates'])
    for key in ('one_step_scores', 'ppr_scores'):
        if set(map(int, scored[key])) != candidates:
            raise ValueError(f'Incomplete {key}')
    for key in ('one_step_ranking', 'ppr_ranking'):
        if len(scored[key]) != 10 or set(scored[key]) != candidates:
            raise ValueError(f'Invalid {key}')


def write_jsonl(path: Path, rows: list[dict]):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')


def ndcg_at_5(position: int) -> float:
    return 1.0 / math.log2(position + 2) if position < 5 else 0.0


def summarize(rows: list[dict], sources: list[dict]) -> dict:
    positions = {'memrec': [], 'one_step': [], 'ppr': []}
    enriched = []
    for scored, source in zip(rows, sources):
        validate_score_row(scored, source)
        target = int(source['target_item'])
        failed = source['failure'] is not None
        current = {'memrec': int(source['target_position'])}
        if not failed and scored['local_ranking'][current['memrec']] != target:
            raise ValueError('Baseline target position changed')
        for arm, key in (('one_step', 'one_step_ranking'), ('ppr', 'ppr_ranking')):
            current[arm] = 10 if failed else scored[key].index(target)
        for arm in positions:
            positions[arm].append(current[arm])
        enriched.append({**scored, 'target_item': target, 'target_positions': current})
    metrics = {}
    for arm, values in positions.items():
        metrics[arm] = {
            'ndcg_at_5': sum(map(ndcg_at_5, values)) / len(values),
            'hit_at_1': sum(value == 0 for value in values) / len(values),
            'hit_at_5': sum(value < 5 for value in values) / len(values),
        }
    for arm in ('one_step', 'ppr'):
        deltas = [ndcg_at_5(a) - ndcg_at_5(b)
                  for a, b in zip(positions[arm], positions['memrec'])]
        metrics[arm]['delta_ndcg_at_5'] = sum(deltas) / len(deltas)
        metrics[arm]['paired_bootstrap_95'] = paired_bootstrap_ci(
            deltas, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        metrics[arm]['improved_worsened_unchanged'] = [
            sum(delta > 0 for delta in deltas), sum(delta < 0 for delta in deltas),
            sum(delta == 0 for delta in deltas),
        ]
    return {'metrics': metrics, 'predictions': enriched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('smoke', 'full'))
    parser.add_argument('--baseline-dir', type=Path, default=BASELINE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    baseline_rows = load_baseline(args.baseline_dir)
    source_hashes = {
        'baseline_predictions': sha256_file(args.baseline_dir / 'test_predictions.jsonl'),
        'dataset': sha256_file(DATA),
        'script': sha256_file(Path(__file__)),
        'graph_primitive': sha256_file(ROOT / 'src/temporal_common/graph.py'),
        'ranking_primitive': sha256_file(ROOT / 'src/temporal_common/metrics.py'),
    }
    dataset = RecDataset(str(DATA), seed=42, precompute_negatives=False)
    graph, graph_stats = build_graph(dataset)
    manifest = {
        'run_id': args.output_dir.name, 'source_sha256': source_hashes,
        'graph_stats': graph_stats, 'graph_source': 'train_data only; per-user ordered positions',
        'seed_source': 'train_data plus validation item, both before test target',
        'candidate_source': 'original InstructRec Books list from frozen MemRec predictions',
        'local_ranker': 'full MemRec 700-warmup/200-dev self-host result',
        'scope': 'post-ranking augmentation; Stage-R/ReRank/Stage-W not rerun',
        'n_eval_users': FULL_USERS, 'smoke_users': SMOKE_USERS,
        'seed_items': SEED_ITEMS, 'restart_probability': RESTART_PROBABILITY,
        'monte_carlo_walks': MONTE_CARLO_WALKS, 'max_walk_steps': MAX_WALK_STEPS,
        'frozen_alpha': ALPHA, 'primary_arm': 'one_step', 'secondary_arm': 'ppr',
        'global_timestamp_available': False, 'gpu_used': False, 'new_llm_requests': 0,
    }
    if args.phase == 'smoke':
        args.output_dir.mkdir(parents=True, exist_ok=False)
        scored = [score_row(row, dataset, graph) for row in baseline_rows[:SMOKE_USERS]]
        for row, source in zip(scored, baseline_rows[:SMOKE_USERS]):
            validate_score_row(row, source)
        if any(row['failure'] is not None for row in scored):
            raise ValueError('Smoke contains a malformed frozen MemRec ranking')
        if scored[0] != score_row(baseline_rows[0], dataset, graph):
            raise ValueError('Transition scoring is not deterministic')
        (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        smoke_file = args.output_dir / 'smoke_scores.jsonl'
        write_jsonl(smoke_file, scored)
        gate = {'status': 'pass', 'n_users': SMOKE_USERS,
                'smoke_scores_sha256': sha256_file(smoke_file),
                'graph_stats': graph_stats, 'new_llm_requests': 0, 'gpu_used': False}
        (args.output_dir / 'smoke_gate.json').write_text(json.dumps(gate, indent=2) + '\n')
        print(json.dumps(gate, sort_keys=True))
        return

    if (args.output_dir / 'full_result.json').exists():
        raise ValueError('Full transition result already exists; refusing overwrite')
    saved_manifest = json.loads((args.output_dir / 'manifest.json').read_text())
    if saved_manifest != manifest:
        raise ValueError('Full transfer contract differs from promoted smoke')
    gate = json.loads((args.output_dir / 'smoke_gate.json').read_text())
    smoke_file = args.output_dir / 'smoke_scores.jsonl'
    if gate['status'] != 'pass' or sha256_file(smoke_file) != gate['smoke_scores_sha256']:
        raise ValueError('Smoke artifact changed or did not pass')
    scored = [json.loads(line) for line in smoke_file.read_text().splitlines()]
    if len(scored) != SMOKE_USERS:
        raise ValueError('Wrong smoke denominator')
    for row, source in zip(scored, baseline_rows[:SMOKE_USERS]):
        validate_score_row(row, source)
    scored.extend(score_row(row, dataset, graph) for row in baseline_rows[SMOKE_USERS:])
    result = summarize(scored, baseline_rows)
    predictions_file = args.output_dir / 'full_predictions.jsonl'
    write_jsonl(predictions_file, result['predictions'])
    report = {
        'status': 'complete', 'n_users': FULL_USERS, 'manifest': manifest,
        'metrics': result['metrics'], 'full_predictions_sha256': sha256_file(predictions_file),
        'baseline_failed_rankings': sum(row['failure'] is not None for row in baseline_rows),
    }
    (args.output_dir / 'full_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'n_users': FULL_USERS,
                      'metrics': report['metrics']}, sort_keys=True))


if __name__ == '__main__':
    main()
