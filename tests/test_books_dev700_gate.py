import json
import sqlite3

from scripts.check_books_memrec_full_dev import check
from src.data.books_protocol import books_cohorts, books_dev_cost_subset, cohort_digest


def test_dev700_gate_requires_locked_cohorts_and_complete_journal(tmp_path):
    cohorts, _ = books_cohorts(list(range(7377)))
    warmup, eval_ids = books_dev_cost_subset(cohorts['all'], cohorts['dev'])
    config = {
        'eval_cohort': 'dev', 'warmup_user_scope': 'subset', 'eval_feedback': 'none',
        'use_pregenerated_candidates': True, 'books_subset_users': 700,
        'memrec': {'reranker_mode': 'llm'}, 'provider': {'revision': 'model-revision'},
    }
    metrics = {
        'n_eval_users': 200, 'n_rankings': 200, 'n_warmup_users': 700,
        'n_stage_r_calls': 200, 'n_stage_rr_calls': 200, 'n_stage_w_calls': 0,
        'n_stage_r_warmup_calls': 700, 'n_stage_rr_warmup_calls': 700,
        'n_stage_w_warmup_calls': 700, 'n_failed_rankings': 0,
        'llm_physical_requests': 2500, 'llm_request_hard_cap': 2750,
        'NDCG@5': 1.0, 'Hit@1': 1.0,
    }
    (tmp_path / 'instructrec-books_memrec_agent_seed42_20000101_000000.json').write_text(
        json.dumps({'config': config, 'test_metrics': metrics})
    )
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'model_revision': 'model-revision', 'eval_cohort': 'dev',
        'n_eval_users': 200, 'warmup_user_scope': 'subset',
        'subset_user_sha256': cohort_digest(warmup),
        'subset_eval_sha256': cohort_digest(eval_ids),
    }))
    rows = ({'user_id': user_id, 'target_item': 0,
             'candidates': list(range(10)), 'ranked_items': list(range(10)),
             'target_position': 0, 'failure': None} for user_id in eval_ids)
    (tmp_path / 'test_predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )
    with sqlite3.connect(tmp_path / 'journal.sqlite') as conn:
        conn.execute('CREATE TABLE entries (phase TEXT, seq INTEGER)')
        conn.executemany('INSERT INTO entries VALUES (?, ?)',
                         [('warmup', seq) for seq in range(700)]
                         + [('eval', seq) for seq in range(200)])
    with sqlite3.connect(tmp_path / 'request-budget.sqlite') as conn:
        conn.execute('CREATE TABLE budget (id INTEGER, lim INTEGER, used INTEGER)')
        conn.execute('INSERT INTO budget VALUES (1, 2750, 2500)')
    assert check(tmp_path, 200, 700, 'subset')['status'] == 'pass'
