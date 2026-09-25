from types import SimpleNamespace

import pytest

from src.data.books_protocol import books_cohorts, books_dev_cost_subset, cohort_digest
from src.memory.storage import MemoryStorage
from src.models.llm_client import DurableRequestBudget, RequestBudgetExceeded
from src.train.books_run_journal import BooksRunJournal
from src.train.trainer_memrec import MemRecTrainer


def test_books_cost_subset_is_fixed_and_all_scored_users_are_warmed():
    cohorts, _ = books_cohorts(list(range(7377)))
    warmup, evaluation = books_dev_cost_subset(cohorts['all'], cohorts['dev'])
    assert len(warmup) == 700
    assert len(evaluation) == 200
    assert evaluation == cohorts['dev'][:200]
    assert set(evaluation).issubset(warmup)
    assert cohort_digest(warmup) == 'e8b9e14032df6e2ea8d0389c62de13d0e8fb087edac127aacbfba8b4062050c9'
    assert cohort_digest(evaluation) == '5395ee7775d9e5d0add11ed386715c34a002138c282f4f910f775ea90871a837'


def test_durable_request_budget_survives_restart(tmp_path):
    path = tmp_path / 'budget.sqlite'
    first = DurableRequestBudget(2, str(path), 'run-a')
    first.consume()
    second = DurableRequestBudget(2, str(path), 'run-a')
    assert second.used == 1
    second.consume()
    with pytest.raises(RequestBudgetExceeded):
        second.consume()
    assert DurableRequestBudget(2, str(path), 'run-a').used == 2
    with pytest.raises(ValueError, match='contract/limit'):
        DurableRequestBudget(2, str(path), 'run-b')


def test_storage_mutations_replay_exactly_once():
    initial = MemoryStorage()
    initial.item_descriptions[7] = 'Book'
    initial.begin_mutation_record()
    initial.update_user_memory(3, 'Reader')
    initial.update_item_memory(7, 'Updated book')
    record = initial.finish_mutation_record()
    restored = MemoryStorage()
    restored.item_descriptions[7] = 'Book'
    restored.apply_mutation_record(record)
    assert restored.user_profiles == initial.user_profiles
    assert restored.item_descriptions == initial.item_descriptions
    assert restored.n_updates == initial.n_updates == 2


def test_warmup_journal_resumes_without_repeating_committed_users(tmp_path):
    path = tmp_path / 'progress.sqlite'

    def make_trainer(fail_after=None):
        trainer = MemRecTrainer.__new__(MemRecTrainer)
        trainer._books_run_journal = BooksRunJournal(str(path), {'contract': 'fixed'})
        trainer.warmup_rounds = 1
        trainer.agent = SimpleNamespace(
            storage=MemoryStorage(), n_stage_r_calls=0,
            n_stage_rr_calls=0, n_stage_w_calls=0,
        )
        trainer.llm_client = SimpleNamespace()
        trainer.reranker_llm_client = None
        called = []

        def run_user(user_id, *_args):
            called.append(user_id)
            if fail_after is not None and len(called) > fail_after:
                raise RuntimeError('simulated interruption')
            trainer.agent.n_stage_r_calls += 1
            trainer.agent.n_stage_rr_calls += 1
            trainer.agent.n_stage_w_calls += 1
            trainer.agent.storage.update_user_memory(user_id, f'profile-{user_id}')
            return True

        trainer._warmup_single_user = run_user
        return trainer, called

    first, called = make_trainer(fail_after=2)
    with pytest.raises(RuntimeError, match='interruption'):
        first._run_training_with_journal([1, 2, 3], 'test')
    assert called == [1, 2, 3]
    assert len(first._books_run_journal.read_phase('warmup', [1, 2, 3])) == 2

    resumed, resumed_calls = make_trainer()
    resumed._run_training_with_journal([1, 2, 3], 'test')
    assert resumed_calls == [3]
    assert resumed.agent.storage.user_profiles == {
        1: 'profile-1', 2: 'profile-2', 3: 'profile-3',
    }
    assert resumed.agent.storage.n_updates == resumed.agent.n_stage_w_calls == 3
    with pytest.raises(ValueError, match='contract'):
        BooksRunJournal(str(path), {'contract': 'changed'})
