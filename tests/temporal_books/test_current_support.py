from src.temporal_books.current_support import (
    event_ndcg_at_5,
    paired_bootstrap_ci,
    rerank_prompt,
    stable_sample_candidates,
)


def test_candidate_sampling_is_fixed_and_excludes_history():
    pool = [*"abcdefghijk", "gold"]
    first = stable_sample_candidates(pool, {"a", "b"}, "gold", n_candidates=10, seed_key="same")
    second = stable_sample_candidates(pool, {"a", "b"}, "gold", n_candidates=10, seed_key="same")
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert "gold" in first
    assert not ({"a", "b"} & set(first))


def test_rerank_schema_is_ranking_only_and_ndcg_is_position_aware():
    event = {"candidate_item_ids": [str(index) for index in range(10)]}
    _, schema = rerank_prompt(event, [], {str(index): "memory" for index in range(10)})
    assert set(schema) == {"ranking"}
    assert event_ndcg_at_5(["a", "b", "c"], "b") > event_ndcg_at_5(["a", "b", "c"], "c")


def test_paired_bootstrap_is_deterministic_and_contains_mean():
    values = [-0.2, 0.0, 0.1, 0.4]
    first = paired_bootstrap_ci(values, resamples=1000, seed=7)
    assert first == paired_bootstrap_ci(values, resamples=1000, seed=7)
    assert first[0] <= sum(values) / len(values) <= first[1]
