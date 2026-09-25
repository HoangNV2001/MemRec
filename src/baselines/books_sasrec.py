"""SASRec on the exact InstructRec Books candidate/cohort protocol.

Training uses every user's pre-test sequence (train + penultimate item), never
the final test target. Architecture/epoch selection uses development users only.
Held-out scoring is intentionally a separate, later phase.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.data import RecDataset
from src.data.books_protocol import BOOKS_CANDIDATE_SHA256, books_cohorts, candidate_digest
from src.temporal_movielens.baseline_models import SASRec, right_padded_sequences
from src.temporal_movielens.baselines import _sasrec_epoch


ROOT = Path(__file__).resolve().parents[2]
BOOKS_INTER = ROOT / 'data/processed/instructrec-books/instructrec-books.inter'
SERVER_RUN_ROOT = Path('/mnt/data/users/anhnct/memrec-hnv/runs')


@dataclass
class BooksSASRecData:
    n_items: int
    pretest_sequences: dict[int, list[int]]
    candidates: dict[int, list[int]]
    targets: dict[int, int]
    dev_users: list[int]
    heldout_users: list[int]
    cohort_manifest: dict[str, Any]
    candidate_sha256: str = ''


def load_books_data() -> BooksSASRecData:
    dataset = RecDataset(str(BOOKS_INTER), seed=42, precompute_negatives=False)
    dataset.load_ranked_lists()
    cohorts, manifest = books_cohorts(list(dataset.test_data))
    if not dataset.ranked_lists or len(dataset.ranked_lists) != len(dataset.test_data):
        raise ValueError('SASRec requires all original Books candidate lists')
    candidates: dict[int, list[int]] = {}
    for uid, target in dataset.test_data.items():
        values = [int(item) for item in dataset.ranked_lists[uid]]
        if len(values) != 10 or len(set(values)) != 10 or target not in values:
            raise ValueError(f'Invalid Books candidates for user {uid}')
        candidates[uid] = values
    digest = candidate_digest(candidates, dataset.test_data)
    if digest != BOOKS_CANDIDATE_SHA256:
        raise ValueError('SASRec Books candidates differ from locked MemRec manifest')
    # Unlike the MemRec train graph, SASRec can train on the penultimate item:
    # that interaction is also fed to MemRec via Stage-W warm-up before test.
    pretest = {
        uid: [*dataset.train_data[uid], dataset.valid_data[uid]]
        for uid in dataset.test_data
    }
    return BooksSASRecData(
        n_items=dataset.n_items,
        pretest_sequences=pretest,
        candidates=candidates,
        targets=dict(dataset.test_data),
        dev_users=cohorts['dev'],
        heldout_users=cohorts['heldout'],
        cohort_manifest=manifest,
        candidate_sha256=digest,
    )


def score_users(
    model: SASRec,
    data: BooksSASRecData,
    user_ids: Sequence[int],
    maximum_sequence_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Return one auditable ranking per user, preserving original tie order."""
    model.eval()
    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(user_ids), 256):
            batch = user_ids[start:start + 256]
            histories, lengths = right_padded_sequences(
                [data.pretest_sequences[uid] for uid in batch], maximum_sequence_length
            )
            candidate_ids = [data.candidates[uid] for uid in batch]
            scores = model.score_candidates(
                histories.to(device),
                lengths.to(device),
                torch.tensor(candidate_ids, dtype=torch.long, device=device) + 1,
            ).detach().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('SASRec produced non-finite candidate scores')
            for uid, values, candidate_row in zip(batch, scores, candidate_ids):
                order = np.argsort(-values, kind='stable').tolist()
                ranked = [candidate_row[index] for index in order]
                output.append({
                    'user_id': uid,
                    'target_item': data.targets[uid],
                    'candidates': candidate_row,
                    'scores': [float(score) for score in values],
                    'ranked_items': ranked,
                    'target_position': ranked.index(data.targets[uid]),
                })
    return output


def ndcg_at_5(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError('Cannot evaluate an empty SASRec cohort')
    return sum(
        1.0 / math.log2(int(row['target_position']) + 2)
        if int(row['target_position']) < 5 else 0.0
        for row in rows
    ) / len(rows)


def ranking_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Use the same one-positive candidate-rank definitions as MemRec."""
    if not rows:
        raise ValueError('Cannot evaluate an empty SASRec cohort')
    positions = [int(row['target_position']) for row in rows]
    return {
        f'{metric}@{k}': sum(
            (position < k) if metric == 'Hit' else
            (1.0 / math.log2(position + 2) if position < k else 0.0)
            for position in positions
        ) / len(positions)
        for k in (1, 3, 5, 10)
        for metric in ('Hit', 'NDCG')
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def locked_source_commit() -> str:
    dirty = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError('SASRec GPU run requires a clean tracked worktree')
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def validate_gpu_output_dir(output_dir: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('SASRec GPU run requires exactly one visible GPU')
    if not output_dir.name.endswith('-hnv'):
        raise ValueError('Output directory must end in -hnv')
    if not output_dir.resolve().is_relative_to(SERVER_RUN_ROOT):
        raise ValueError(f'SASRec GPU outputs must be under {SERVER_RUN_ROOT}')


def selected_dev_predictions(
    checkpoint: Path,
    data: BooksSASRecData,
    config: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Reload the winning checkpoint and verify its locked evaluation contract."""
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    architecture = payload['architecture']
    if architecture not in config['architectures'] or payload['common_config'] != dict(config):
        raise ValueError('SASRec checkpoint configuration differs from the locked grid')
    if payload['dev_cohort_sha256'] != data.cohort_manifest['cohort_sha256']['dev']:
        raise ValueError('SASRec checkpoint development cohort differs from this dataset')
    if payload['candidate_sha256'] != data.candidate_sha256:
        raise ValueError('SASRec checkpoint candidate manifest differs from this dataset')
    model = make_model(data.n_items, config, architecture).to(device)
    model.load_state_dict(payload['state_dict'])
    rows = score_users(model, data, data.dev_users, int(architecture['maximum_sequence_length']), device)
    del model
    return rows


def make_model(n_items: int, common: Mapping[str, Any], architecture: Mapping[str, Any]) -> SASRec:
    return SASRec(
        n_items=n_items,
        embedding_dim=int(architecture['embedding_dim']),
        maximum_sequence_length=int(architecture['maximum_sequence_length']),
        transformer_blocks=int(common['transformer_blocks']),
        attention_heads=int(common['attention_heads']),
        dropout=float(architecture['dropout']),
    )


def cpu_smoke(data: BooksSASRecData, config: Mapping[str, Any]) -> dict[str, Any]:
    """One 30-user training batch and 30 fixed-list scores, with no GPU."""
    users = data.dev_users[:30]
    settings = config['architectures'][0]
    torch.manual_seed(int(config['seed']))
    model = make_model(data.n_items, config, settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    loss = _sasrec_epoch(
        model,
        {uid: data.pretest_sequences[uid] for uid in users},
        range(data.n_items),
        optimizer,
        30,
        int(settings['maximum_sequence_length']),
        np.random.default_rng(int(config['seed'])),
        torch.device('cpu'),
    )
    rows = score_users(model, data, users, int(settings['maximum_sequence_length']), torch.device('cpu'))
    if not math.isfinite(loss) or len(rows) != 30:
        raise RuntimeError('SASRec CPU smoke failed')
    if any(len(row['ranked_items']) != 10 or set(row['ranked_items']) != set(row['candidates']) for row in rows):
        raise RuntimeError('SASRec CPU smoke produced malformed rankings')
    return {'users': 30, 'finite_loss': True, 'valid_rankings': 30, 'gpu_used': False}


def gpu_smoke(data: BooksSASRecData, config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """One-batch/30-user smoke for every predeclared architecture, before full train."""
    validate_gpu_output_dir(output_dir)
    source_commit = locked_source_commit()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing SASRec run: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    users = data.dev_users[:30]
    device = torch.device('cuda:0')
    results = []
    for architecture in config['architectures']:
        seed = int(config['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
        model = make_model(data.n_items, config, architecture).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
        )
        loss = _sasrec_epoch(
            model, {uid: data.pretest_sequences[uid] for uid in users},
            range(data.n_items), optimizer, 30,
            int(architecture['maximum_sequence_length']),
            np.random.default_rng(seed), device,
        )
        rows = score_users(model, data, users, int(architecture['maximum_sequence_length']), device)
        if not math.isfinite(loss) or len(rows) != 30:
            raise RuntimeError(f"SASRec GPU smoke failed for {architecture['id']}")
        if any(len(row['ranked_items']) != 10 or set(row['ranked_items']) != set(row['candidates']) for row in rows):
            raise RuntimeError(f"SASRec GPU smoke malformed ranking for {architecture['id']}")
        results.append({
            'architecture_id': architecture['id'], 'users': 30,
            'finite_loss': True, 'valid_rankings': 30,
            'peak_vram_bytes': torch.cuda.max_memory_allocated(device),
        })
        del model, optimizer
        torch.cuda.empty_cache()
    total_vram = torch.cuda.get_device_properties(device).total_memory
    if any(row['peak_vram_bytes'] > total_vram for row in results):
        raise RuntimeError('SASRec GPU smoke exceeded one visible card')
    manifest = {
        'source_commit': source_commit,
        'config_sha256': config_digest(config),
        'candidate_sha256': data.candidate_sha256,
        'dev_cohort_sha256': data.cohort_manifest['cohort_sha256']['dev'],
        'smoke_user_ids': users,
        'gpu_count': torch.cuda.device_count(),
        'gpu_total_vram_bytes': total_vram,
        'architectures': results,
        'passed': len(results) == len(config['architectures']),
    }
    (output_dir / 'gpu_smoke-hnv.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    return manifest


def train_dev(data: BooksSASRecData, config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Predeclared grid + early stopping on dev; never scores held-out users."""
    validate_gpu_output_dir(output_dir)
    source_commit = locked_source_commit()
    smoke_file = output_dir / 'gpu_smoke-hnv.json'
    if not smoke_file.is_file():
        raise FileNotFoundError('Full SASRec dev training requires prior 30-user GPU smoke')
    if {path.name for path in output_dir.iterdir()} != {smoke_file.name}:
        raise FileExistsError(f'Refusing to reuse a non-smoke SASRec run: {output_dir}')
    smoke = json.loads(smoke_file.read_text(encoding='utf-8'))
    if (
        smoke.get('passed') is not True
        or smoke.get('source_commit') != source_commit
        or smoke.get('config_sha256') != config_digest(config)
        or smoke.get('candidate_sha256') != data.candidate_sha256
        or smoke.get('dev_cohort_sha256') != data.cohort_manifest['cohort_sha256']['dev']
        or smoke.get('smoke_user_ids') != data.dev_users[:30]
        or smoke.get('gpu_count') != 1
        or [item['architecture_id'] for item in smoke.get('architectures', [])]
        != [item['id'] for item in config['architectures']]
    ):
        raise ValueError('SASRec GPU smoke manifest does not match the full training contract')

    device = torch.device('cuda:0')
    seed = int(config['seed'])
    runs: list[dict[str, Any]] = []
    for architecture in config['architectures']:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = make_model(data.n_items, config, architecture).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
        )
        rng = np.random.default_rng(seed)
        best = -math.inf
        best_epoch = 0
        stale = 0
        history = []
        checkpoint = output_dir / f"{architecture['id']}-hnv.pt"
        for epoch in range(1, int(config['maximum_epochs']) + 1):
            loss = _sasrec_epoch(
                model, data.pretest_sequences, range(data.n_items), optimizer,
                int(config['batch_size']), int(architecture['maximum_sequence_length']),
                rng, device,
            )
            if not math.isfinite(loss):
                raise RuntimeError(f"SASRec non-finite train loss for {architecture['id']} epoch {epoch}")
            dev_rows = score_users(
                model, data, data.dev_users,
                int(architecture['maximum_sequence_length']), device,
            )
            score = ndcg_at_5(dev_rows)
            history.append({'epoch': epoch, 'train_loss': loss, 'dev_ndcg_at_5': score})
            if score > best:
                best = score
                best_epoch = epoch
                stale = 0
                torch.save({
                    'state_dict': model.state_dict(),
                    'architecture': dict(architecture),
                    'common_config': dict(config),
                    'source_commit': source_commit,
                    'dev_cohort_sha256': data.cohort_manifest['cohort_sha256']['dev'],
                    'candidate_sha256': data.candidate_sha256,
                    'selected_epoch': best_epoch,
                }, checkpoint)
            else:
                stale += 1
            if stale >= int(config['early_stopping_patience']):
                break
        runs.append({
            'architecture_id': architecture['id'],
            'best_epoch': best_epoch,
            'best_dev_ndcg_at_5': best,
            'training_log': history,
            'checkpoint': str(checkpoint),
        })
        del model, optimizer
        torch.cuda.empty_cache()

    winner = max(range(len(runs)), key=lambda index: (runs[index]['best_dev_ndcg_at_5'], -index))
    selected_checkpoint = Path(runs[winner]['checkpoint'])
    dev_rows = selected_dev_predictions(selected_checkpoint, data, config, device)
    dev_metrics = ranking_metrics(dev_rows)
    if not math.isclose(dev_metrics['NDCG@5'], runs[winner]['best_dev_ndcg_at_5'], abs_tol=1e-7):
        raise RuntimeError('Reloaded SASRec checkpoint changed the selected dev metric')
    predictions_path = output_dir / 'dev_predictions-hnv.jsonl'
    with predictions_path.open('w', encoding='utf-8') as handle:
        for row in dev_rows:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    result = {
        'source_commit': source_commit,
        'dev_users': len(data.dev_users),
        'dev_cohort_sha256': data.cohort_manifest['cohort_sha256']['dev'],
        'candidate_sha256': data.candidate_sha256,
        'heldout_users_scored': 0,
        'manual_tuning_performed': False,
        'selection_metric': 'NDCG@5 on locked dev cohort',
        'grid_results': runs,
        'selected_architecture_id': runs[winner]['architecture_id'],
        'selected_checkpoint': str(selected_checkpoint),
        'selected_checkpoint_sha256': sha256_file(selected_checkpoint),
        'dev_predictions': str(predictions_path),
        'dev_predictions_sha256': sha256_file(predictions_path),
        'dev_metrics': dev_metrics,
    }
    (output_dir / 'dev_selection-hnv.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return result
