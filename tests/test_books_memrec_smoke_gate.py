import json

import pytest

from scripts.check_books_memrec_smoke import check


def fixture_run(tmp_path):
    config = {
        'eval_cohort': 'dev', 'warmup_user_scope': 'eval', 'eval_feedback': 'none',
        'use_pregenerated_candidates': True,
        'memrec': {'reranker_mode': 'llm'}, 'provider': {'revision': 'pinned-sha'},
    }
    metrics = {
        'n_eval_users': 30, 'n_rankings': 30, 'n_warmup_users': 30,
        'n_failed_rankings': 0, 'n_stage_r_warmup_calls': 30,
        'n_stage_rr_warmup_calls': 30, 'n_stage_w_warmup_calls': 30,
        'n_stage_r_calls': 30, 'n_stage_rr_calls': 30, 'n_stage_w_calls': 0,
        'llm_physical_requests': 150, 'llm_request_hard_cap': 165,
    }
    result = tmp_path / 'instructrec-books_memrec_agent_seed42_20000101_000000.json'
    result.write_text(json.dumps({'config': config, 'test_metrics': metrics}))
    rows = [
        {'user_id': uid, 'target_item': 0, 'candidates': list(range(10)),
         'ranked_items': list(range(10)), 'target_position': 0, 'failure': None}
        for uid in range(30)
    ]
    (tmp_path / 'test_predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )
    return result, rows


def test_smoke_gate_passes_complete_full_memrec_run(tmp_path):
    fixture_run(tmp_path)
    assert check(tmp_path)['physical_requests'] == 150


def test_smoke_gate_rejects_partial_ranking(tmp_path):
    _, rows = fixture_run(tmp_path)
    rows[0]['ranked_items'] = list(range(9))
    (tmp_path / 'test_predictions.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows)
    )
    with pytest.raises(ValueError, match='Incomplete permutation'):
        check(tmp_path)


def test_smoke_gate_rejects_missing_stage_w(tmp_path):
    result, _ = fixture_run(tmp_path)
    payload = json.loads(result.read_text())
    payload['test_metrics']['n_stage_w_warmup_calls'] = 0
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Stage-W'):
        check(tmp_path)


def test_cache_backed_subset_smoke_requires_matching_promoted_predictions(tmp_path):
    result, _ = fixture_run(tmp_path)
    payload = json.loads(result.read_text())
    payload['config']['books_subset_users'] = 700
    payload['test_metrics']['llm_physical_requests'] = 0
    result.write_text(json.dumps(payload))
    reference = tmp_path / 'promoted_predictions.jsonl'
    predictions = tmp_path / 'test_predictions.jsonl'
    reference.write_bytes(predictions.read_bytes())
    assert check(tmp_path, allow_cache=True, reference_predictions=reference)['physical_requests'] == 0
    with pytest.raises(ValueError, match='physical'):
        check(tmp_path)
    reference.write_text(reference.read_text().replace('"target_position": 0',
                                                        '"target_position": 1', 1))
    with pytest.raises(ValueError, match='differs'):
        check(tmp_path, allow_cache=True, reference_predictions=reference)
