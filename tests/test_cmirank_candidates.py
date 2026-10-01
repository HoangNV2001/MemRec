import json
import csv
from pathlib import Path

import numpy as np
import pytest

from src.cmirank.candidates import (
    MixedCandidateSampler, digest_key, item_text, popularity_bucket,
    popularity_from_snapshot,
    validate_candidate_contract,
)
from src.cmirank.policy_inputs import read_train_histories
from src.cmirank.metadata import read_metadata_texts


def sampler():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(30, 6)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return MixedCandidateSampler(range(30), vectors, {i: i % 5 for i in range(30)},
                                 seed="candidate-test")


def test_mixed_sampler_is_reproducible_unique_and_prefix_safe():
    model = sampler()
    first = model.sample("user-1-pseudo", 10, [1, 2, 3], additionally_forbidden=[4])
    assert first == model.sample("user-1-pseudo", 10, [3, 2, 1], additionally_forbidden=[4])
    assert len(first.candidate_ids) == len(set(first.candidate_ids)) == 10
    assert first.candidate_ids.count(10) == 1
    assert not set(first.candidate_ids) & {1, 2, 3, 4}
    assert sorted(map(len, first.negative_components.values())) == [3, 3, 3]
    negatives = [item for group in first.negative_components.values() for item in group]
    assert len(set(negatives)) == 9 and 10 not in negatives


@pytest.mark.parametrize("field,value", [
    ("composition", {"uniform": 9}), ("eligible_pool", "test_candidates"),
    ("popularity_bucket", "full_data"), ("shuffle", "positive_last"),
])
def test_contract_change_cannot_silently_relabel_the_sampler(field, value):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cmirank/candidate_sampler_v1.json").read_text())
    validate_candidate_contract(config)
    config[field] = value
    with pytest.raises(ValueError, match="Unsupported"):
        validate_candidate_contract(config)


def test_contract_rejects_fake_semantic_encoder_or_changed_template():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/cmirank/candidate_sampler_v1.json").read_text())
    config["encoder"]["pooling"] = "python_hash"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_candidate_contract(config)


def test_policy_input_has_no_sampling_or_reward_side_channel():
    row = sampler().sample("u1", 10, [0])
    policy = row.policy_input()
    assert set(policy) == {"episode_id", "candidate_ids"}
    assert "positive" not in json.dumps(policy)
    assert "semantic" not in json.dumps(policy)
    assert row.reward_audit()["positive_item_id"] == 10


def test_semantic_component_is_top_three_of_remaining_pool():
    model = sampler()
    result = model.sample("u2", 10, [0, 1])
    blocked = {0, 1, 10, *result.negative_components["uniform"],
               *result.negative_components["popularity_matched"]}
    scores = model.vectors @ model.vectors[10]
    expected = sorted((i for i in range(30) if i not in blocked),
                      key=lambda i: (-float(scores[i]),
                                     digest_key(model.seed, "u2", "semantic_tie", i), i))[:3]
    assert result.negative_components["semantic_hard"] == tuple(expected)


def test_popularity_count_and_nearest_bucket_fallback_use_only_snapshot():
    assert popularity_from_snapshot({1: (3, 3), 2: (4, 3)}) == {3: 3, 4: 1}
    assert [popularity_bucket(i) for i in (0, 1, 2, 3, 7)] == [0, 1, 1, 2, 3]
    model = sampler()
    model.popularity = {10: 1000}  # positive's bucket is unavailable in negatives
    result = model.sample("nearest", 10, [0])
    blocked = {0, 10, *result.negative_components["uniform"]}
    nearest = sorted((i for i in range(30) if i not in blocked), key=lambda i: (
        abs(int(model.buckets[i]) - popularity_bucket(1000)), int(model.buckets[i]),
        digest_key(model.seed, "nearest", "popularity", i), i))[:3]
    assert result.negative_components["popularity_matched"] == tuple(nearest)


def test_semantic_ties_have_deterministic_hash_breaking():
    vectors = np.ones((30, 1), dtype=np.float32)
    model = MixedCandidateSampler(range(30), vectors, {}, seed="tie")
    row = model.sample("u1", 0, [1])
    blocked = {0, 1, *row.negative_components["uniform"],
               *row.negative_components["popularity_matched"]}
    expected = sorted((i for i in range(30) if i not in blocked),
                      key=lambda i: (digest_key("tie", "u1", "semantic_tie", i), i))[:3]
    assert row.negative_components["semantic_hard"] == tuple(expected)


@pytest.mark.parametrize("case", ["duplicates", "unsorted", "nonfinite", "nonunit", "dtype"])
def test_sampler_rejects_incomplete_or_corrupt_index(case):
    ids = list(range(30))
    vectors = sampler().vectors.copy()
    if case == "duplicates": ids[1] = 0
    if case == "unsorted": ids.reverse()
    if case == "nonfinite": vectors[0, 0] = np.nan
    if case == "nonunit": vectors *= 2
    if case == "dtype": vectors = vectors.astype(np.float64)
    with pytest.raises(ValueError):
        MixedCandidateSampler(ids, vectors, {}, seed="test")


def test_sampler_rejects_missing_metadata_history_positive_and_small_pool():
    model = sampler()
    for positive, prefix in ((99, [1]), (10, [10]), (10, list(range(22)))):
        with pytest.raises(ValueError): model.sample("u1", positive, prefix)
    with pytest.raises(ValueError, match="Not enough"):
        model.sample("u1", 25, list(range(22)))


def test_item_text_contains_only_bounded_static_metadata():
    assert item_text(" A  Book ", " a\nsummary ", max_characters=100) == (
        "Title: A Book\nDescription: a summary")
    assert len(item_text("A", "x" * 1000, max_characters=32)) == 32
    with pytest.raises(ValueError): item_text(" ", "abc", max_characters=100)


def test_train_adapter_discards_suffix_identities_and_sorts_by_order(tmp_path):
    source = tmp_path / "fixture.inter"
    source.write_text("user_id\titem_id\ttimestamp\n"
                      "1\tSEALED_TEST\t4\n1\t7\t1\n1\t8\t2\n"
                      "1\tSEALED_VALIDATION\t3\n", encoding="utf-8")
    assert read_train_histories(source) == {1: (7, 8)}


def test_train_adapter_rejects_ambiguous_order(tmp_path):
    source = tmp_path / "fixture.inter"
    source.write_text("user_id\titem_id\ttimestamp\n1\t5\t1\n1\t6\t1\n1\t7\t2\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        read_train_histories(source)


def test_identical_metadata_duplicates_collapse_without_changing_text(tmp_path):
    source = tmp_path / "fixture.meta"
    source.write_text("item_id\tasin\ttitle\tdescription\n1\tA1\tBook\tSummary\n"
                      "1\tA1\tBook\tSummary\n2\tA2\t\tNo title\n")
    texts, audit = read_metadata_texts(source, max_characters=100)
    assert texts == {1: "Title: Book\nDescription: Summary"}
    assert audit["metadata_rows"] == 3
    assert audit["metadata_unique_ids"] == 2
    assert audit["identical_duplicate_rows_collapsed"] == 1
    assert audit["empty_title_items_excluded"] == 1


@pytest.mark.parametrize("field", ["asin", "title", "description"])
def test_conflicting_metadata_is_not_silently_last_row_wins(tmp_path, field):
    source = tmp_path / "fixture.meta"
    first = {"item_id": 1, "asin": "A1", "title": "Book", "description": "Summary"}
    second = {**first, field: "Different"}
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first), delimiter="\t")
        writer.writeheader()
        writer.writerows([first, second])
    with pytest.raises(ValueError, match="Conflicting"):
        read_metadata_texts(source, max_characters=100)
