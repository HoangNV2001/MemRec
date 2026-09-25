"""Protocol guard for the original InstructRec test candidate lists."""

from types import SimpleNamespace

import pytest

from src.train.trainer_memrec import MemRecTrainer
from src.data.dataset_base import RecDataset
from src.data.books_protocol import books_cohorts, cohort_digest, split_cohorts
from src.models.reranker_llm import LLMReranker
from src.models.llm_client import (
    LLMClient, RequestBudget, RequestBudgetExceeded, validate_json_shape,
)
from src.memory.graph import UserItemGraph
from scripts.run_train import redact_credentials
from src.utils import load_config


def trainer_with_candidates(candidates, n_items=20):
    trainer = MemRecTrainer.__new__(MemRecTrainer)
    trainer.use_pregenerated_candidates = True
    trainer.n_eval_candidates = 10
    trainer.dataset = SimpleNamespace(ranked_lists={0: candidates}, n_items=n_items)
    return trainer


def test_fixed_candidates_preserve_original_order():
    original = [9, 1, 7, 2, 8, 3, 6, 4, 5, 0]
    trainer = trainer_with_candidates(original)
    assert trainer._evaluation_candidates(0, 7, 'test') == original
    assert trainer._evaluation_candidates(0, 7, 'test') is not original


@pytest.mark.parametrize('candidates,target', [
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 8], 8),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8], 8),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 20], 8),
])
def test_fixed_candidates_reject_invalid_sets(candidates, target):
    trainer = trainer_with_candidates(candidates)
    with pytest.raises(ValueError):
        trainer._evaluation_candidates(0, target, 'test')


def test_fixed_candidates_reject_validation_without_its_own_list():
    trainer = trainer_with_candidates(list(range(10)))
    with pytest.raises(ValueError, match='test-only'):
        trainer._evaluation_candidates(0, 0, 'valid')
    with pytest.raises(ValueError, match='test-only'):
        trainer.evaluate(split='valid')
    with pytest.raises(ValueError, match='serial'):
        trainer.evaluate(split='test', parallel=True)
    trainer.eval_feedback = 'gt'
    with pytest.raises(ValueError, match='test-time feedback'):
        trainer.evaluate(split='test')


def test_full_benchmark_cannot_accidentally_use_partial_warmup():
    trainer = trainer_with_candidates(list(range(10)))
    trainer.eval_feedback = 'none'
    trainer.debug = False
    trainer.eval_cohort = None
    trainer.eval_user_list = None
    trainer.n_eval_users = None
    trainer.warmup_user_scope = 'eval'
    trainer.dataset.test_data = {uid: 0 for uid in range(31)}
    with pytest.raises(ValueError, match='Partial warm-up'):
        trainer.evaluate(split='test')


def test_result_config_redacts_provider_credentials():
    config = {'provider': {'api_key': 'do-not-persist', 'model': 'same-model'}}
    clean = redact_credentials(config)
    assert clean['provider'] == {'api_key': '[REDACTED]', 'model': 'same-model'}
    assert config['provider']['api_key'] == 'do-not-persist'


def test_lazy_negatives_and_deterministic_warmup(tmp_path):
    data_path = tmp_path / 'tiny.inter'
    data_path.write_text(
        'user_id\titem_id\ttimestamp\n'
        '0\t0\t1\n0\t1\t2\n0\t2\t3\n0\t3\t4\n'
        '1\t4\t1\n1\t5\t2\n1\t6\t3\n1\t19\t4\n'
    )
    dataset = RecDataset(str(data_path), precompute_negatives=False)
    assert dataset.user_negatives == {}
    assert set(dataset.sample_negative_items(0, 10)).isdisjoint({0, 1, 2, 3})
    graph = UserItemGraph(dataset)
    assert graph.get_user_items(0) == [0, 1]
    assert 2 not in graph.users_by_item and 3 not in graph.users_by_item
    assert dataset.get_user_history(0, split='test')[-2] == dataset.valid_data[0]

    trainer = MemRecTrainer.__new__(MemRecTrainer)
    trainer.use_pregenerated_candidates = True
    trainer.n_eval_candidates = 10
    trainer.dataset = dataset
    trainer.config = {'seed': 42}
    first = trainer._construct_candidates(0, 2, [0, 1])
    second = trainer._construct_candidates(0, 2, [0, 1])
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert set(first).isdisjoint({0, 1, 3})


def test_full_benchmark_config_disables_test_feedback(monkeypatch):
    monkeypatch.setenv('MEMREC_SELFHOST_MODEL', 'local-checkpoint')
    monkeypatch.setenv('MEMREC_SELFHOST_REVISION', 'pinned-revision')
    monkeypatch.setenv('MEMREC_SELFHOST_BASE_URL', 'http://localhost:8000/v1')
    monkeypatch.setenv('MEMREC_SELFHOST_API_KEY', 'local-placeholder')
    config = load_config('configs/memrec_instructrec-books_full_benchmark.yaml')
    assert config['use_pregenerated_candidates'] is True
    assert config['eval_feedback'] == 'none'
    assert config['warmup']['enabled'] is True
    assert config['memrec']['reranker_mode'] == 'llm'
    assert config['provider']['model'] == 'local-checkpoint'
    assert config['provider']['revision'] == 'pinned-revision'
    assert config['eval_cohort'] == 'dev'
    assert config['memrec']['upstream_empty_facets_prompt'] is True


def test_books_cohort_smoke_is_disjoint_and_deterministic():
    users = list(range(30))
    exposed = {0, 3, 7, 11, 17, 23}
    first = split_cohorts(users, exposed, dev_size=12, seed='smoke-v1')
    second = split_cohorts(users, exposed, dev_size=12, seed='smoke-v1')
    assert first == second
    assert len(first['dev']) == 12
    assert len(first['heldout']) == 18
    assert exposed.issubset(first['dev'])
    assert set(first['dev']).isdisjoint(first['heldout'])
    assert sorted(first['dev'] + first['heldout']) == users
    assert len(cohort_digest(first['heldout'])) == 64


def test_books_full_cohort_hash_is_locked():
    cohorts, manifest = books_cohorts(list(range(7377)))
    assert len(cohorts['dev']) == 2000
    assert len(cohorts['heldout']) == 5377
    assert manifest['historically_exposed_users'] == 1085
    assert manifest['cohort_sha256']['heldout'] == (
        'b7751e0ae8f4f23709736ed9b19c457e570aa3edf2a6e0db19c6da938c4af8be'
    )


def test_upstream_empty_facets_prompt_is_explicit():
    reranker = LLMReranker(None)
    kwargs = {'user_id': 0, 'facets': [], 'candidates': [{'id': 7, 'title': 'Book'}]}
    upstream = reranker.build_rerank_prompt(**kwargs, upstream_empty_facets_prompt=True)[0]['content']
    fork = reranker.build_rerank_prompt(**kwargs)[0]['content']
    assert '**User Preferences (Extracted from Collaborative Memories):**' in upstream
    assert '(No facets extracted)' in upstream
    assert '(No facets extracted)' not in fork


def test_benchmark_position_counts_malformed_and_fallback_as_misses():
    candidates = list(range(10))
    position = MemRecTrainer._benchmark_position
    assert position(7, candidates, candidates, {'rerank_scores': []}) == (7, None)
    assert position(7, candidates, candidates[:-1], {}) == (10, 'malformed_ranking')
    fallback = {'rerank_scores': [{'rationale': 'Error'} for _ in candidates]}
    assert position(7, candidates, candidates, fallback) == (10, 'llm_fallback')
    invalid = {'rerank_scores': [{'score': 1.2, 'rationale': 'bad'}]}
    assert position(7, candidates, candidates, invalid) == (10, 'invalid_score')


def test_physical_request_budget_fails_before_second_network_call():
    calls = []
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))],
        usage=None,
    )
    client = LLMClient.__new__(LLMClient)
    client.model = client.request_model = 'cpu-fake'
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs) or completion))
    )
    client.request_budget = RequestBudget(1)
    client.total_physical_requests = 0
    client._log_conversation = lambda **kwargs: None
    assert client.generate([{'role': 'user', 'content': 'x'}]) == '{}'
    with pytest.raises(RequestBudgetExceeded):
        client.generate([{'role': 'user', 'content': 'x'}])
    assert len(calls) == client.total_physical_requests == 1


def test_strict_llm_schema_rejects_missing_nonfinite_and_extra_fields():
    schema = {
        'type': 'object',
        'properties': {
            'scores': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {'item_id': {'type': 'integer'}, 'score': {'type': 'number'}},
                    'required': ['item_id', 'score'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['scores'],
        'additionalProperties': False,
    }
    validate_json_shape({'scores': [{'item_id': 7, 'score': 0.5}]}, schema)
    for invalid in ({}, {'scores': [{'item_id': 7, 'score': float('nan')}]},
                    {'scores': [{'item_id': 7, 'score': 0.5, 'extra': 'x'}]}):
        with pytest.raises(ValueError):
            validate_json_shape(invalid, schema)
