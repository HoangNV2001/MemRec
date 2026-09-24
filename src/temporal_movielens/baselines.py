"""No-search MostPopular, BPR-MF and SASRec baselines for MovieLens M7."""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from src.temporal_books.common import (
    PROJECT_ROOT,
    jsonl_write,
    load_yaml,
    project_path,
    sha256_file,
    stable_order,
)
from src.temporal_books.current_support import event_key, stable_sample_candidates
from src.temporal_common.metrics import rank_by_scores
from src.temporal_movielens.baseline_models import BPRMF, SASRec, right_padded_sequences
from src.temporal_movielens.data import (
    eligible_singleton_targets,
    iter_user_ratings,
    load_movies,
    positive_candidate_pool,
    select_one_target_per_user,
    strict_history,
)
from src.temporal_movielens.graph import load_locked_inputs
from src.temporal_movielens.prepare import verify_audit


SCHEMA_VERSION = 1
VALIDATION_K = 10


@dataclass
class EncodedInteractions:
    item_ids: list[str]
    item_index: Dict[str, int]
    user_ids: list[str]
    user_index: Dict[str, int]
    train_sequences: Dict[int, list[int]]
    final_sequences: Dict[int, list[int]]
    candidate_indices: list[int]
    popularity: np.ndarray


@dataclass
class ValidationEvent:
    user_index: int
    history: list[int]
    candidates: list[int]
    gold_index: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_source_commit() -> str:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("baseline GPU run requires a clean tracked worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _device(require_cuda: bool) -> torch.device:
    if require_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("baseline GPU phase requires exactly one visible CUDA device")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_prepared(config: Mapping[str, Any], config_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    prepared, method_lock, manifest = load_locked_inputs(config)
    if manifest.get("config", {}).get("sha256") != sha256_file(config_path):
        raise RuntimeError("M7 baseline config changed after prepare")
    if method_lock.get("manual_tuning_performed") is not False:
        raise RuntimeError("M7 method lock permits manual tuning")
    if method_lock.get("baselines") != config["baselines"]:
        raise RuntimeError("baseline config differs from method lock")
    return prepared, manifest


def load_encoded_interactions(config: Mapping[str, Any]) -> EncodedInteractions:
    audit = verify_audit(config)
    dataset, baselines = config["dataset"], config["baselines"]
    train_cutoff = int(audit["temporal_split"]["train_cutoff"])
    final_cutoff = int(audit["temporal_split"]["validation_cutoff"])
    rating_min = float(baselines["positive_rating_min"])
    movies = load_movies(project_path(dataset["movies_file"]))
    item_ids = sorted(movies, key=int)
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    raw: Dict[str, tuple[list[str], list[str]]] = {}
    all_users: list[str] = []
    popularity: Counter[str] = Counter()
    for user_id, events in iter_user_ratings(project_path(dataset["ratings_file"])):
        all_users.append(user_id)
        train = [
            item_id
            for timestamp, item_id, rating in events
            if timestamp < train_cutoff and rating >= rating_min
        ]
        final = [
            item_id
            for timestamp, item_id, rating in events
            if timestamp < final_cutoff and rating >= rating_min
        ]
        if final:
            raw[user_id] = (train, final)
            popularity.update(final)
    user_ids = sorted(all_users, key=int)
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    train_sequences = {
        user_index[user_id]: [item_index[item_id] for item_id in values[0]]
        for user_id, values in raw.items()
        if values[0]
    }
    final_sequences = {
        user_index[user_id]: [item_index[item_id] for item_id in values[1]]
        for user_id, values in raw.items()
        if values[1]
    }
    pool = positive_candidate_pool(
        project_path(dataset["ratings_file"]),
        train_cutoff=train_cutoff,
        positive_rating_min=rating_min,
    )
    candidate_indices = [item_index[item_id] for item_id in pool]
    popularity_array = np.zeros(len(item_ids), dtype=np.float32)
    for item_id, count in popularity.items():
        popularity_array[item_index[item_id]] = float(count)
    return EncodedInteractions(
        item_ids=item_ids,
        item_index=item_index,
        user_ids=user_ids,
        user_index=user_index,
        train_sequences=train_sequences,
        final_sequences=final_sequences,
        candidate_indices=candidate_indices,
        popularity=popularity_array,
    )


def build_validation_events(
    config: Mapping[str, Any], encoded: EncodedInteractions
) -> list[ValidationEvent]:
    audit = verify_audit(config)
    dataset, study, baselines = config["dataset"], config["study"], config["baselines"]
    ratings_path = project_path(dataset["ratings_file"])
    pool = set(
        positive_candidate_pool(
            ratings_path,
            train_cutoff=int(audit["temporal_split"]["train_cutoff"]),
            positive_rating_min=float(baselines["positive_rating_min"]),
        )
    )
    candidates_per_event = int(baselines["validation_candidates"])
    rows: list[ValidationEvent] = []
    for user_id, events in iter_user_ratings(ratings_path):
        targets = eligible_singleton_targets(
            user_id,
            events,
            start=int(audit["temporal_split"]["train_cutoff"]),
            end=int(audit["temporal_split"]["validation_cutoff"]),
            candidate_pool=pool,
            session_gap_seconds=int(study["session_gap_seconds"]),
            positive_rating_min=float(baselines["positive_rating_min"]),
            minimum_positive_history=int(study["minimum_positive_history"]),
        )
        if not targets or user_id not in encoded.user_index:
            continue
        target = select_one_target_per_user(
            targets, salt=str(baselines["validation_salt"]) + "-target"
        )
        history_events = strict_history(events, int(target["timestamp"]))
        known = {item_id for _, item_id, _ in history_events}
        gold = str(target["gold_item_id"])
        candidate_ids = stable_sample_candidates(
            sorted(pool, key=int),
            known,
            gold,
            n_candidates=candidates_per_event,
            seed_key=f"{baselines['validation_salt']}\0{user_id}\0{target['timestamp']}\0{gold}",
        )
        history = [
            encoded.item_index[item_id]
            for _, item_id, rating in history_events
            if rating >= float(baselines["positive_rating_min"]) and item_id in encoded.item_index
        ]
        if not history:
            continue
        rows.append(
            ValidationEvent(
                user_index=encoded.user_index[user_id],
                history=history,
                candidates=[encoded.item_index[item_id] for item_id in candidate_ids],
                gold_index=encoded.item_index[gold],
            )
        )
    rows.sort(
        key=lambda row: stable_order(
            f"{baselines['validation_salt']}\0{row.user_index}\0{row.gold_index}"
        )
    )
    if not rows:
        raise RuntimeError("no baseline validation events")
    return rows


def _negative(
    rng: np.random.Generator, universe: np.ndarray, positives: set[int]
) -> int:
    while True:
        value = int(universe[int(rng.integers(len(universe)))])
        if value not in positives:
            return value


def _bpr_epoch(
    model: BPRMF,
    sequences: Mapping[int, Sequence[int]] | tuple[np.ndarray, np.ndarray],
    candidate_indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if isinstance(sequences, tuple):
        pair_users, pair_items = sequences
    else:
        count = sum(len(items) for items in sequences.values())
        pair_users = np.fromiter(
            (user for user, items in sequences.items() for _ in items),
            dtype=np.int32,
            count=count,
        )
        pair_items = np.fromiter(
            (item for items in sequences.values() for item in items),
            dtype=np.int32,
            count=count,
        )
    order = rng.permutation(len(pair_users))
    universe = np.asarray(candidate_indices, dtype=np.int64)
    if not len(pair_users) or not len(universe):
        raise ValueError("BPR training data and candidate universe must be non-empty")
    key_modulus = max(int(pair_items.max(initial=0)), int(universe.max(initial=0))) + 1
    positive_keys = np.unique(pair_users.astype(np.int64) * key_modulus + pair_items)

    def known_positive(keys: np.ndarray) -> np.ndarray:
        # np.isin(keys, positive_keys) repeatedly scans/sorts the full 32M-scale
        # interaction vector. The sorted-key lookup keeps each rejection check
        # O(batch log interactions) without materialising Python user-item sets.
        positions = np.searchsorted(positive_keys, keys)
        valid = positions < len(positive_keys)
        result = np.zeros(len(keys), dtype=bool)
        result[valid] = positive_keys[positions[valid]] == keys[valid]
        return result

    total = batches = 0
    model.train()
    for start in range(0, len(order), batch_size):
        selected = order[start : start + batch_size]
        users = pair_users[selected].astype(np.int64, copy=False)
        positives = pair_items[selected].astype(np.int64, copy=False)
        negatives = universe[rng.integers(len(universe), size=len(selected))]
        invalid = known_positive(users * key_modulus + negatives)
        while np.any(invalid):
            negatives[invalid] = universe[rng.integers(len(universe), size=int(invalid.sum()))]
            invalid = known_positive(users * key_modulus + negatives)
        optimizer.zero_grad(set_to_none=True)
        loss = model.pairwise_loss(
            torch.from_numpy(users).to(device),
            torch.from_numpy(positives).to(device),
            torch.from_numpy(negatives).to(device),
        )
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(1, batches)


def _sasrec_batch(
    user_sequences: Sequence[Sequence[int]],
    maximum_sequence_length: int,
    universe: np.ndarray,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.zeros((len(user_sequences), maximum_sequence_length), dtype=torch.long)
    positives = torch.zeros_like(inputs)
    negatives = torch.zeros_like(inputs)
    for row, full in enumerate(user_sequences):
        values = list(full[-(maximum_sequence_length + 1) :])
        source, target = values[:-1], values[1:]
        positive_set = set(full)
        length = len(source)
        if length == 0:
            continue
        inputs[row, :length] = torch.tensor([value + 1 for value in source])
        positives[row, :length] = torch.tensor([value + 1 for value in target])
        negatives[row, :length] = torch.tensor(
            [_negative(rng, universe, positive_set) + 1 for _ in target]
        )
    return inputs, positives, negatives


def _sasrec_epoch(
    model: SASRec,
    sequences: Mapping[int, Sequence[int]],
    candidate_indices: Sequence[int],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    maximum_sequence_length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    usable = [values for values in sequences.values() if len(values) >= 2]
    if not usable or not candidate_indices:
        raise ValueError("SASRec training data and candidate universe must be non-empty")
    order = rng.permutation(len(usable))
    universe = np.asarray(candidate_indices, dtype=np.int64)
    total = batches = 0
    model.train()
    for start in range(0, len(order), batch_size):
        values = [usable[int(index)] for index in order[start : start + batch_size]]
        inputs, positives, negatives = _sasrec_batch(
            values, maximum_sequence_length, universe, rng
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.pointwise_loss(
            inputs.to(device), positives.to(device), negatives.to(device)
        )
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(1, batches)


def _mean_validation_ndcg(
    model: torch.nn.Module,
    name: str,
    events: Sequence[ValidationEvent],
    config: Mapping[str, Any],
    device: torch.device,
) -> float:
    batch_size = 512
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(events), batch_size):
            batch = events[start : start + batch_size]
            candidate_tensor = torch.tensor([row.candidates for row in batch], device=device)
            if name == "bpr_mf":
                users = torch.tensor([row.user_index for row in batch], device=device)
                scores = model.score_candidates(users, candidate_tensor)
            else:
                maximum = int(config["baselines"]["sasrec"]["maximum_sequence_length"])
                sequences, lengths = right_padded_sequences(
                    [row.history for row in batch], maximum
                )
                scores = model.score_candidates(
                    sequences.to(device), lengths.to(device), candidate_tensor + 1
                )
            for row, score_values in zip(batch, scores.detach().cpu().tolist()):
                string_candidates = [str(item) for item in row.candidates]
                score_map = {
                    item: float(score) for item, score in zip(string_candidates, score_values)
                }
                ranking = rank_by_scores(string_candidates, score_map)
                try:
                    rank = ranking.index(str(row.gold_index)) + 1
                except ValueError:
                    rank = len(ranking) + 1
                values.append(1.0 / math.log2(rank + 1) if rank <= VALIDATION_K else 0.0)
    return sum(values) / len(values)


def _fit_with_validation(
    name: str,
    encoded: EncodedInteractions,
    validation: Sequence[ValidationEvent],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    settings = config["baselines"][name]
    seed = int(settings["seed"])

    def make_model() -> torch.nn.Module:
        _seed_everything(seed)
        if name == "bpr_mf":
            return BPRMF(
                len(encoded.user_ids), len(encoded.item_ids), int(settings["embedding_dim"])
            ).to(device)
        return SASRec(
            n_items=len(encoded.item_ids),
            embedding_dim=int(settings["embedding_dim"]),
            maximum_sequence_length=int(settings["maximum_sequence_length"]),
            transformer_blocks=int(settings["transformer_blocks"]),
            attention_heads=int(settings["attention_heads"]),
            dropout=float(settings["dropout"]),
        ).to(device)

    model = make_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["l2"])
    )
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_metric = float("-inf")
    stale = 0
    log: list[Dict[str, float | int]] = []
    train_bpr_pairs: tuple[np.ndarray, np.ndarray] | None = None
    if name == "bpr_mf":
        count = sum(len(items) for items in encoded.train_sequences.values())
        train_bpr_pairs = (
            np.fromiter(
                (user for user, items in encoded.train_sequences.items() for _ in items),
                dtype=np.int32,
                count=count,
            ),
            np.fromiter(
                (item for items in encoded.train_sequences.values() for item in items),
                dtype=np.int32,
                count=count,
            ),
        )
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        if name == "bpr_mf":
            loss = _bpr_epoch(
                model,
                train_bpr_pairs,
                encoded.candidate_indices,
                optimizer,
                int(settings["batch_size"]),
                rng,
                device,
            )
        else:
            loss = _sasrec_epoch(
                model,
                encoded.train_sequences,
                encoded.candidate_indices,
                optimizer,
                int(settings["batch_size"]),
                int(settings["maximum_sequence_length"]),
                rng,
                device,
            )
        metric = _mean_validation_ndcg(model, name, validation, config, device)
        log.append({"epoch": epoch, "train_loss": loss, "validation_ndcg_at_10": metric})
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["early_stopping_patience"]):
            break
    if best_epoch <= 0:
        raise RuntimeError(f"{name} failed to select an epoch")

    # Refit from the same initialization for the selected epoch count on every
    # positive interaction strictly before the final graph/test cutoff.
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    final_model = make_model()
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["l2"]),
    )
    final_rng = np.random.default_rng(seed)
    final_losses: list[float] = []
    final_bpr_pairs: tuple[np.ndarray, np.ndarray] | None = None
    if name == "bpr_mf":
        count = sum(len(items) for items in encoded.final_sequences.values())
        final_bpr_pairs = (
            np.fromiter(
                (user for user, items in encoded.final_sequences.items() for _ in items),
                dtype=np.int32,
                count=count,
            ),
            np.fromiter(
                (item for items in encoded.final_sequences.values() for item in items),
                dtype=np.int32,
                count=count,
            ),
        )
    for _ in range(best_epoch):
        if name == "bpr_mf":
            final_losses.append(
                _bpr_epoch(
                    final_model,
                    final_bpr_pairs,
                    encoded.candidate_indices,
                    final_optimizer,
                    int(settings["batch_size"]),
                    final_rng,
                    device,
                )
            )
        else:
            final_losses.append(
                _sasrec_epoch(
                    final_model,
                    encoded.final_sequences,
                    encoded.candidate_indices,
                    final_optimizer,
                    int(settings["batch_size"]),
                    int(settings["maximum_sequence_length"]),
                    final_rng,
                    device,
                )
            )
    return final_model, {
        "selected_epoch": best_epoch,
        "best_validation_ndcg_at_10": best_metric,
        "selection_log": log,
        "refit_epochs": best_epoch,
        "refit_losses": final_losses,
        "manual_tuning_performed": False,
    }


def _blind_event_rows(
    prepared: Mapping[str, Any], encoded: EncodedInteractions, config: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    by_user = {str(event["user_id"]): event for event in prepared["primary_events"]}
    positive_histories: Dict[str, list[int]] = {}
    for user_id, ratings in iter_user_ratings(project_path(config["dataset"]["ratings_file"])):
        event = by_user.get(user_id)
        if event is None:
            continue
        history = strict_history(ratings, int(event["timestamp"]))
        positive_histories[user_id] = [
            encoded.item_index[item_id]
            for _, item_id, rating in history
            if rating >= float(config["baselines"]["positive_rating_min"])
        ]
    rows: list[Dict[str, Any]] = []
    for event in prepared["primary_events"]:
        user_id = str(event["user_id"])
        if user_id not in encoded.user_index:
            raise RuntimeError(f"baseline user missing from encoding: {user_id}")
        candidates = [str(value) for value in event["candidate_item_ids"]]
        history = positive_histories.get(user_id, [])
        if not history:
            raise RuntimeError(f"SASRec positive history missing: {event_key(event)}")
        rows.append(
            {
                "event_key": event_key(event),
                "user_index": encoded.user_index[user_id],
                "candidate_item_ids": candidates,
                "candidate_indices": [encoded.item_index[item_id] for item_id in candidates],
                "history_indices": history,
            }
        )
    return rows


def _score_model(
    name: str,
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> list[Dict[str, float]]:
    output: list[Dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            batch = rows[start : start + 512]
            candidate_indices = torch.tensor(
                [row["candidate_indices"] for row in batch], device=device
            )
            if name == "bpr_mf":
                users = torch.tensor([row["user_index"] for row in batch], device=device)
                values = model.score_candidates(users, candidate_indices)
            else:
                maximum = int(config["baselines"]["sasrec"]["maximum_sequence_length"])
                sequences, lengths = right_padded_sequences(
                    [row["history_indices"] for row in batch], maximum
                )
                values = model.score_candidates(
                    sequences.to(device), lengths.to(device), candidate_indices + 1
                )
            for row, scores in zip(batch, values.detach().cpu().tolist()):
                output.append(
                    {
                        item_id: float(score)
                        for item_id, score in zip(row["candidate_item_ids"], scores)
                    }
                )
    return output


def offline_smoke(config: Mapping[str, Any]) -> Dict[str, Any]:
    device = _device(False)
    _seed_everything(7)
    bpr = BPRMF(4, 12, int(config["baselines"]["bpr_mf"]["embedding_dim"])).to(device)
    bpr_optimizer = torch.optim.Adam(bpr.parameters(), lr=0.001)
    bpr_loss = bpr.pairwise_loss(
        torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 4])
    )
    bpr_loss.backward()
    bpr_optimizer.step()
    sas_settings = config["baselines"]["sasrec"]
    sasrec = SASRec(
        12,
        int(sas_settings["embedding_dim"]),
        int(sas_settings["maximum_sequence_length"]),
        int(sas_settings["transformer_blocks"]),
        int(sas_settings["attention_heads"]),
        float(sas_settings["dropout"]),
    ).to(device)
    sas_optimizer = torch.optim.Adam(sasrec.parameters(), lr=0.001)
    inputs = torch.tensor([[1, 2, 3], [2, 4, 0]])
    positives = torch.tensor([[2, 3, 4], [4, 5, 0]])
    negatives = torch.tensor([[8, 9, 10], [9, 10, 0]])
    sas_loss = sasrec.pointwise_loss(inputs, positives, negatives)
    sas_loss.backward()
    sas_optimizer.step()
    return {
        "decision": "pass",
        "bpr_loss_finite": math.isfinite(float(bpr_loss.detach())),
        "sasrec_loss_finite": math.isfinite(float(sas_loss.detach())),
        "artifact_written": False,
        "gpu_used": False,
        "outcomes_evaluated": False,
    }


def real_gpu_smoke(
    config: Mapping[str, Any], config_path: Path
) -> Dict[str, Any]:
    prepared, manifest = _load_prepared(config, config_path)
    output = project_path(config["baselines"]["smoke_manifest"])
    if output.exists():
        raise FileExistsError(output)
    source_commit = _clean_source_commit()
    device = _device(True)
    smoke_count = int(config["baselines"]["smoke_events"])
    events = sorted(
        prepared["primary_events"],
        key=lambda event: stable_order(f"m7-baseline-smoke\0{event_key(event)}"),
    )[:smoke_count]
    user_ids = {str(event["user_id"]) for event in events}
    events_by_user = {str(event["user_id"]): event for event in events}
    movies = load_movies(project_path(config["dataset"]["movies_file"]))
    item_ids = sorted(movies, key=int)
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    user_index = {user_id: index for index, user_id in enumerate(sorted(user_ids, key=int))}
    sequences: Dict[int, list[int]] = {}
    inference_histories: Dict[str, list[int]] = {}
    audit = verify_audit(config)
    cutoff = int(audit["temporal_split"]["validation_cutoff"])
    for user_id, ratings in iter_user_ratings(project_path(config["dataset"]["ratings_file"])):
        if user_id not in user_ids:
            continue
        event = events_by_user[user_id]
        positives = [
            item_index[item_id]
            for timestamp, item_id, rating in ratings
            if timestamp < cutoff and rating >= float(config["baselines"]["positive_rating_min"])
        ]
        if positives:
            sequences[user_index[user_id]] = positives
        inference_histories[user_id] = [
            item_index[item_id]
            for timestamp, item_id, rating in ratings
            if timestamp < int(event["timestamp"])
            and rating >= float(config["baselines"]["positive_rating_min"])
        ]
    pool = positive_candidate_pool(
        project_path(config["dataset"]["ratings_file"]),
        train_cutoff=int(audit["temporal_split"]["train_cutoff"]),
        positive_rating_min=float(config["baselines"]["positive_rating_min"]),
    )
    universe = [item_index[item_id] for item_id in pool]
    if not sequences or sum(len(values) >= 2 for values in sequences.values()) == 0:
        raise RuntimeError("baseline smoke has no usable training sequences")
    if len(inference_histories) != smoke_count or any(
        not values for values in inference_histories.values()
    ):
        raise RuntimeError("baseline smoke inference histories missing")
    rows = [
        {
            "event_key": event_key(event),
            "user_index": user_index[str(event["user_id"])],
            "candidate_item_ids": [str(item_id) for item_id in event["candidate_item_ids"]],
            "candidate_indices": [
                item_index[str(item_id)] for item_id in event["candidate_item_ids"]
            ],
            "history_indices": inference_histories[str(event["user_id"])],
        }
        for event in events
    ]
    # PyTorch 2.10 on the cluster rejects peak-memory telemetry before the
    # CUDA context exists, even though is_available/device_count have passed.
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    bpr_settings = config["baselines"]["bpr_mf"]
    _seed_everything(int(bpr_settings["seed"]))
    bpr = BPRMF(len(user_index), len(item_ids), int(bpr_settings["embedding_dim"])).to(device)
    bpr_optimizer = torch.optim.Adam(bpr.parameters(), lr=float(bpr_settings["learning_rate"]))
    bpr_loss = _bpr_epoch(
        bpr,
        sequences,
        universe,
        bpr_optimizer,
        int(bpr_settings["batch_size"]),
        np.random.default_rng(int(bpr_settings["seed"])),
        device,
    )
    bpr_scores = _score_model("bpr_mf", bpr, rows, config, device)
    del bpr, bpr_optimizer
    torch.cuda.empty_cache()
    sas_settings = config["baselines"]["sasrec"]
    _seed_everything(int(sas_settings["seed"]))
    sasrec = SASRec(
        len(item_ids),
        int(sas_settings["embedding_dim"]),
        int(sas_settings["maximum_sequence_length"]),
        int(sas_settings["transformer_blocks"]),
        int(sas_settings["attention_heads"]),
        float(sas_settings["dropout"]),
    ).to(device)
    sas_optimizer = torch.optim.Adam(sasrec.parameters(), lr=float(sas_settings["learning_rate"]))
    sas_loss = _sasrec_epoch(
        sasrec,
        sequences,
        universe,
        sas_optimizer,
        int(sas_settings["batch_size"]),
        int(sas_settings["maximum_sequence_length"]),
        np.random.default_rng(int(sas_settings["seed"])),
        device,
    )
    sasrec_scores = _score_model("sasrec", sasrec, rows, config, device)
    scored_values = [
        value
        for score_rows in (bpr_scores, sasrec_scores)
        for score_row in score_rows
        for value in score_row.values()
    ]
    if len(scored_values) != smoke_count * int(config["study"]["candidates_per_event"]) * 2:
        raise RuntimeError("baseline smoke score count mismatch")
    if not all(math.isfinite(value) for value in scored_values):
        raise RuntimeError("baseline smoke produced non-finite scores")
    peak = torch.cuda.max_memory_allocated(0) / (1024**3)
    del sasrec, sas_optimizer
    torch.cuda.empty_cache()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "source_commit": source_commit,
        "prepared_sha256": manifest["prepared"]["sha256"],
        "method_lock_sha256": manifest["method_lock"]["sha256"],
        "events": smoke_count,
        "models": ["bpr_mf", "sasrec"],
        "bpr_loss": bpr_loss,
        "sasrec_loss": sas_loss,
        "finite_candidate_scores": len(scored_values),
        "peak_vram_gib": peak,
        "cuda_device": torch.cuda.get_device_name(0),
        "manual_tuning_performed": False,
        "recommendation_outcomes_evaluated": False,
        "test_labels_used_for_training_or_scoring": False,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def verify_gpu_smoke(config: Mapping[str, Any], config_path: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    path = project_path(config["baselines"]["smoke_manifest"])
    if not path.exists():
        raise RuntimeError("baseline full run blocked: GPU smoke manifest missing")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "decision": "pass",
        "config_sha256": sha256_file(config_path),
        "source_commit": _clean_source_commit(),
        "prepared_sha256": manifest["prepared"]["sha256"],
        "method_lock_sha256": manifest["method_lock"]["sha256"],
        "events": int(config["baselines"]["smoke_events"]),
        "manual_tuning_performed": False,
        "recommendation_outcomes_evaluated": False,
        "test_labels_used_for_training_or_scoring": False,
    }
    for key, value in expected.items():
        if smoke.get(key) != value:
            raise RuntimeError(f"baseline smoke mismatch: {key}")
    return smoke


def run_full(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    prepared, prepare_manifest = _load_prepared(config, config_path)
    smoke = verify_gpu_smoke(config, config_path, prepare_manifest)
    scores_path = project_path(config["baselines"]["scores"])
    manifest_path = project_path(config["baselines"]["manifest"])
    if scores_path.exists() or manifest_path.exists():
        raise FileExistsError("baseline score/manifest exists")
    device = _device(True)
    encoded = load_encoded_interactions(config)
    validation = build_validation_events(config, encoded)
    blind = _blind_event_rows(prepared, encoded, config)
    baseline_scores: Dict[str, list[Dict[str, float]]] = {
        "most_popular": [
            {
                item_id: float(encoded.popularity[item_index])
                for item_id, item_index in zip(row["candidate_item_ids"], row["candidate_indices"])
            }
            for row in blind
        ]
    }
    training: Dict[str, Any] = {}
    checkpoint_hashes: Dict[str, str] = {}
    for name in ("bpr_mf", "sasrec"):
        model, training_log = _fit_with_validation(name, encoded, validation, config, device)
        baseline_scores[name] = _score_model(name, model, blind, config, device)
        checkpoint = manifest_path.with_name(f"m7_{name}_checkpoint-hnv.pt")
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint
        )
        checkpoint_hashes[name] = sha256_file(checkpoint)
        training[name] = training_log
        del model
        torch.cuda.empty_cache()
    output_rows = []
    for index, row in enumerate(blind):
        output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_key": row["event_key"],
                "candidate_item_ids": row["candidate_item_ids"],
                "scores": {name: values[index] for name, values in baseline_scores.items()},
                "test_labels_used_for_scoring": False,
            }
        )
    records, digest = jsonl_write(scores_path, output_rows)
    peak = torch.cuda.max_memory_allocated(0) / (1024**3)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["study"]["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "complete",
        "config_sha256": sha256_file(config_path),
        "source_commit": smoke["source_commit"],
        "prepared_sha256": prepare_manifest["prepared"]["sha256"],
        "method_lock_sha256": prepare_manifest["method_lock"]["sha256"],
        "smoke_manifest_sha256": sha256_file(project_path(config["baselines"]["smoke_manifest"])),
        "smoke_peak_vram_gib": smoke["peak_vram_gib"],
        "full_peak_vram_gib": peak,
        "validation_events": len(validation),
        "training": training,
        "checkpoints_sha256": checkpoint_hashes,
        "scores": {"path": config["baselines"]["scores"], "records": records, "sha256": digest},
        "manual_tuning_performed": False,
        "recommendation_outcomes_evaluated": False,
        "test_labels_used_for_training_or_scoring": False,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens M7 preregistered classical baselines")
    parser.add_argument(
        "--config", default="configs/temporal_movielens32m/m7_graph_hard_end2end.yaml"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--offline-smoke", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--request", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_yaml(config_path)
    if args.offline_smoke:
        result = offline_smoke(config)
    elif args.smoke:
        result = real_gpu_smoke(config, config_path)
    else:
        result = run_full(config, config_path)
    printable = dict(result)
    if "training" in printable:
        printable["training"] = {
            name: {
                "selected_epoch": values["selected_epoch"],
                "best_validation_ndcg_at_10": values["best_validation_ndcg_at_10"],
            }
            for name, values in printable["training"].items()
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
