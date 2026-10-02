"""Frozen CPU-only dataset probes, not a recommendation method or policy input."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .candidates import shuffle_candidates
from .metadata import read_metadata_texts
from .provenance import file_sha256


PROBES = ("position", "item_id", "popularity", "title_length", "metadata_length",
          "description_present", "semantic_centrality")
METRICS = ("hit_at_1", "hit_at_5", "ndcg_at_5")


def validate_audit_contract(config: dict) -> None:
    fixed = {"schema_version": 1, "probes": list(PROBES), "permutations": 9999,
             "seed": 20261002, "familywise_alpha": 0.05, "material_hit1_delta": 0.05,
             "fit": "position_frequencies_or_score_direction_from_policy_train_only",
             "direction_objective": "train_ndcg_at_5_positive_direction_on_tie",
             "ties": "exact_expected_credit_over_all_tied_ranks_no_id_or_position_tiebreak",
             "semantic_centrality": "mean_off_diagonal_cosine_within_candidate_set_no_target_anchor",
             "null": "independent_uniform_target_position_per_validation_user",
             "multiple_tests": "bonferroni_seven_predeclared_validation_probes",
             "phase": "pseudo_only_warmup_integrity_checked_separately",
             "smoke_users": 20, "training_ready": False}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("Unsupported shortcut audit contract")


def verify_candidate_run(run_dir: Path, config: dict, inputs: dict, recipes: dict) -> list[dict]:
    """Recheck completed full rows, frozen roles, smoke reuse and CPU cleanup."""
    full_dir, smoke_dir = run_dir / "full-hnv", run_dir / "smoke-hnv"
    if (file_sha256(full_dir / "report.json") != config["full_report_sha256"]
            or file_sha256(full_dir / "candidate-rows.jsonl") != config["full_candidates_sha256"]
            or (run_dir / "source-commit.txt").read_text().strip() != config["candidate_source_commit"]):
        raise ValueError("Candidate run does not match pinned completed artifacts")
    cleanup = json.loads((run_dir / "cleanup.json").read_text())
    if cleanup != {"process_exit_code": 0, "device": "cpu",
                   "gpu_requested": False, "child_exited": True}:
        raise ValueError("Candidate task has no successful CPU cleanup")
    full, smoke = [json.loads((directory / "report.json").read_text())
                   for directory in (full_dir, smoke_dir)]
    expected = {"source_commit": config["candidate_source_commit"],
                "graph_snapshot_sha256": inputs["snapshot_sha256"],
                "policy_split_manifest_sha256": inputs["manifest_sha256"],
                "episode_contract_sha256": recipes["episode_contract_sha256"],
                "candidate_contract_sha256": recipes["candidate_contract_sha256"],
                "index_manifest_sha256": recipes["index_manifest_sha256"],
                "original_test_candidates_accessed": False,
                "original_instruction_accessed": False,
                "original_suffix_item_ids_parsed_or_used": False,
                "negative_history_collisions": 0, "warmup_own_pseudo_target_negatives": 0,
                "real_llm_requests": 0, "training_ready": False}
    for report in (full, smoke):
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError("Candidate provenance differs")
    eligible = sorted(inputs["warmups"])
    if (full["status"] != "COMPLETE_APPROVED_POLICY_CANDIDATES_NOT_G0_OR_PPO_PROMOTION"
            or full["query_user_ids"] != eligible or full["candidate_sets"] != 2 * len(eligible)
            or full["stage_counts"] != {"warmup": len(eligible), "pseudo": len(eligible)}
            or full["smoke_candidate_sets_reused"] != 40
            or smoke["status"] != "CANDIDATE_INTEGRITY_SMOKE_PASS_NOT_G0_OR_RANKING_RESULT"
            or smoke["users"] != 20 or smoke["candidate_sets"] != 40
            or file_sha256(smoke_dir / "candidate-rows.jsonl") != smoke["candidate_manifest_sha256"]):
        raise ValueError("Incomplete full candidate scope or smoke gate")
    lines = (full_dir / "candidate-rows.jsonl").read_bytes().splitlines(keepends=True)
    rows, raw_by_episode, positions = [], {}, {kind: Counter() for kind in ("warmup", "pseudo")}
    expected_order = [(uid, kind) for uid in eligible for kind in ("warmup", "pseudo")]
    if len(lines) != len(expected_order):
        raise ValueError("Wrong full candidate row count")
    for line, (uid, kind) in zip(lines, expected_order):
        row = json.loads(line)
        if set(row) != {"policy_input", "reward_audit", "provenance"}:
            raise ValueError("Unexpected candidate row fields")
        policy, reward = row["policy_input"], row["reward_audit"]
        episode = f"books-policy-v1-user-{uid}-{kind}"
        ids, positive = policy["candidate_ids"], inputs["warmups" if kind == "warmup" else "targets"][uid]
        components = reward["negative_components"]
        negatives = [item for items in components.values() for item in items]
        forbidden = set(inputs["snapshot"].train_data[uid])
        forbidden.add(inputs["targets" if kind == "warmup" else "warmups"][uid])
        seed = recipes["episode_config"]["warmup"]["seed"] if kind == "warmup" else recipes["config"]["seed"]
        if (set(policy) != {"episode_id", "candidate_ids"} or policy["episode_id"] != episode
                or len(ids) != 10 or any(type(item) is not int for item in ids)
                or len(set(ids)) != 10 or ids.count(positive) != 1 or positive in forbidden
                or not set(ids) <= recipes["catalog_ids"]
                or {key: len(value) for key, value in components.items()}
                != recipes["episode_config"][kind]["composition"]
                or len(set(negatives)) != 9 or set(negatives) != set(ids) - {positive}
                or set(negatives) & forbidden or not set(negatives) <= recipes["pseudo"].positions.keys()
                or reward["positive_item_id"] != positive or reward["positive_position"] != ids.index(positive)
                or set(reward) != {"positive_item_id", "positive_position", "negative_components"}
                or tuple(ids) != shuffle_candidates(ids, seed=seed, episode_id=episode)
                or row["provenance"] != {"user_id": uid, "kind": kind,
                    "snapshot_sha256": inputs["snapshot_sha256"],
                    "episode_contract_sha256": recipes["episode_contract_sha256"]}):
            raise ValueError("Candidate role/identity/order/input integrity failed")
        rows.append(row)
        raw_by_episode[episode] = line
        positions[kind][str(ids.index(positive))] += 1
    smoke_lines = (smoke_dir / "candidate-rows.jsonl").read_bytes().splitlines(keepends=True)
    if len(smoke_lines) != 40 or len({json.loads(line)["policy_input"]["episode_id"] for line in smoke_lines}) != 40:
        raise ValueError("Wrong smoke row count or duplicate smoke episode")
    if any(raw_by_episode.get(json.loads(line)["policy_input"]["episode_id"]) != line for line in smoke_lines):
        raise ValueError("Full rows did not reuse smoke bytes")
    if any(dict(positions[kind]) != full["positive_position_counts"][kind] for kind in positions):
        raise ValueError("Position ledger differs from rows")
    return rows


def rank_credits(scores: np.ndarray) -> np.ndarray:
    """Expected Hit@1/5, NDCG@5 for every possible target, averaging exact ties."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 10 or not np.isfinite(scores).all():
        raise ValueError("Need finite scores for ten candidates per episode")
    low = 1 + (scores[:, None, :] > scores[:, :, None]).sum(axis=2)
    high = (scores[:, None, :] >= scores[:, :, None]).sum(axis=2)
    count = high - low + 1
    gains = np.zeros(10)
    gains[:5] = 1 / np.log2(np.arange(1, 6) + 1)
    prefix = np.r_[0.0, np.cumsum(gains)]
    return np.stack(((low == 1) / count, np.maximum(0, np.minimum(high, 5) - low + 1) / count,
                     (prefix[high] - prefix[low - 1]) / count), axis=2)


def fit_probe_scores(features: dict[str, np.ndarray], targets: np.ndarray,
                     train_mask: np.ndarray) -> tuple[dict, dict]:
    """Only train labels determine position frequencies or +/- direction."""
    if not train_mask.any() or set(features) != set(PROBES):
        raise ValueError("Need all fixed probes and a nonempty train group")
    chosen, scores = {}, {}
    rows = np.arange(train_mask.sum())
    train_targets = targets[train_mask]
    for name in PROBES:
        values = features[name]
        if name == "position":
            counts = np.bincount(train_targets, minlength=10)
            chosen[name] = {"train_position_counts": counts.tolist()}
            scores[name] = np.tile(counts, (len(targets), 1))
        else:
            objectives = [float(rank_credits(sign * values[train_mask])[rows, train_targets, 2].mean())
                          for sign in (1, -1)]
            sign = 1 if objectives[0] >= objectives[1] else -1
            chosen[name] = {"direction": sign, "train_ndcg_positive_negative": objectives}
            scores[name] = sign * values
    return scores, chosen


def null_pvalue(hit1_credits: np.ndarray, observed: float, *, seed: int, permutations: int) -> float:
    """Uniform random targets conditional on fixed, target-blind probe scores."""
    if hit1_credits.ndim != 2 or hit1_credits.shape[1] != 10 or len(hit1_credits) == 0:
        raise ValueError("Need a nonempty ten-slot validation group")
    rng, exceed = np.random.default_rng(seed), 0
    for start in range(0, permutations, 256):
        draws = rng.integers(0, 10, size=(min(256, permutations - start), len(hit1_credits)))
        means = hit1_credits[np.arange(len(hit1_credits))[None, :], draws].mean(axis=1)
        exceed += int((means >= observed - 1e-12).sum())
    return (1 + exceed) / (permutations + 1)


def feature_rows(rows: list[dict], recipes: dict, texts: dict[int, str],
                 lengths: dict[int, tuple[int, int, bool]]) -> list[dict]:
    """No reward, target identity or negative-component labels read here."""
    sampler = recipes["pseudo"]
    result = []
    for row in rows:
        policy = row["policy_input"]
        ids = policy["candidate_ids"]
        vectors = sampler.vectors[[sampler.positions[item] for item in ids]].astype(np.float64)
        gram = vectors @ vectors.T
        values = {"position": list(range(10)), "item_id": ids,
                  "popularity": [sampler.popularity.get(item, 0) for item in ids],
                  "title_length": [lengths[item][0] for item in ids],
                  "metadata_length": [len(texts[item]) for item in ids],
                  "description_present": [int(lengths[item][2]) for item in ids],
                  "semantic_centrality": ((gram.sum(axis=1) - np.diag(gram)) / 9).tolist()}
        feature = {"episode_id": policy["episode_id"], "candidate_ids": ids, "features": values}
        validate_feature_row(feature, policy)
        result.append(feature)
    return result


def validate_feature_row(feature: dict, policy: dict) -> None:
    if (set(feature) != {"episode_id", "candidate_ids", "features"}
            or feature["episode_id"] != policy["episode_id"]
            or feature["candidate_ids"] != policy["candidate_ids"]
            or set(feature["features"]) != set(PROBES)
            or any(np.asarray(values).shape != (10,) or not np.isfinite(values).all()
                   for values in feature["features"].values())):
        raise ValueError("Invalid target-blind audit feature row")


def metadata_features(source: Path, *, max_characters: int) -> tuple[dict, dict]:
    """Static sampler text lengths; identical duplicates/conflicts audited first."""
    import csv
    texts, _ = read_metadata_texts(source, max_characters=max_characters)
    lengths = {}
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = int(row["item_id"])
            if item not in lengths and item in texts:
                title, description = " ".join(row["title"].split()), " ".join(row["description"].split())
                lengths[item] = (len(title), len(description), bool(description))
    return texts, lengths
