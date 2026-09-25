"""SASRec adapter must use exact candidate rows and pre-test histories."""

import math

import pytest
import torch

from src.baselines.books_sasrec import (
    BooksSASRecData, cpu_smoke, make_model, ndcg_at_5,
    ranking_metrics, score_users, selected_dev_predictions, train_dev,
)


def tiny_books_data():
    histories = {uid: [uid % 10, (uid + 1) % 10, (uid + 2) % 10] for uid in range(30)}
    candidates = {uid: list(range(10, 20)) for uid in range(30)}
    targets = {uid: 10 + uid % 10 for uid in range(30)}
    return BooksSASRecData(
        n_items=40,
        pretest_sequences=histories,
        candidates=candidates,
        targets=targets,
        dev_users=list(range(30)),
        heldout_users=[],
        cohort_manifest={'cohort_sha256': {'dev': 'synthetic'}},
    )


def test_sasrec_scores_preserve_fixed_candidates():
    data = tiny_books_data()
    common = {'transformer_blocks': 2, 'attention_heads': 2}
    architecture = {'embedding_dim': 16, 'maximum_sequence_length': 10, 'dropout': 0.2}
    model = make_model(data.n_items, common, architecture)
    rows = score_users(model, data, data.dev_users[:3], 10, torch.device('cpu'))
    assert len(rows) == 3
    for row in rows:
        assert row['candidates'] == list(range(10, 20))
        assert sorted(row['ranked_items']) == row['candidates']
        assert 0 <= row['target_position'] < 10
        assert all(math.isfinite(value) for value in row['scores'])


def test_sasrec_30_user_cpu_smoke_has_finite_loss():
    data = tiny_books_data()
    config = {
        'seed': 7, 'learning_rate': 0.001,
        'transformer_blocks': 2, 'attention_heads': 2,
        'architectures': [
            {'id': 'tiny', 'embedding_dim': 16, 'maximum_sequence_length': 10, 'dropout': 0.2}
        ],
    }
    result = cpu_smoke(data, config)
    assert result == {'users': 30, 'finite_loss': True, 'valid_rankings': 30, 'gpu_used': False}


def test_sasrec_ndcg_at_5_denominator_includes_misses():
    rows = [{'target_position': 0}, {'target_position': 9}]
    assert ndcg_at_5(rows) == 0.5
    assert ranking_metrics(rows)['NDCG@5'] == 0.5
    assert ranking_metrics(rows)['Hit@10'] == 1.0


def test_sasrec_checkpoint_reload_checks_cohort_and_candidate(tmp_path):
    data = tiny_books_data()
    config = {
        'transformer_blocks': 2, 'attention_heads': 2,
        'architectures': [{'id': 'tiny', 'embedding_dim': 16,
                           'maximum_sequence_length': 10, 'dropout': 0.2}],
    }
    model = make_model(data.n_items, config, config['architectures'][0])
    checkpoint = tmp_path / 'sasrec.pt'
    payload = {
        'state_dict': model.state_dict(),
        'architecture': config['architectures'][0],
        'common_config': config,
        'dev_cohort_sha256': 'synthetic',
        'candidate_sha256': '',
    }
    torch.save(payload, checkpoint)
    rows = selected_dev_predictions(checkpoint, data, config, torch.device('cpu'))
    assert len(rows) == 30
    payload['candidate_sha256'] = 'different'
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='candidate manifest'):
        selected_dev_predictions(checkpoint, data, config, torch.device('cpu'))


def test_sasrec_full_train_requires_matching_gpu_smoke(tmp_path, monkeypatch):
    import src.baselines.books_sasrec as module
    monkeypatch.setattr(module, 'SERVER_RUN_ROOT', tmp_path)
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(module.torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(module, 'locked_source_commit', lambda: 'fixed-sha')
    data = tiny_books_data()
    config = {
        'seed': 7, 'learning_rate': 0.001, 'weight_decay': 0.0,
        'transformer_blocks': 2, 'attention_heads': 2,
        'architectures': [{'id': 'tiny', 'embedding_dim': 16,
                           'maximum_sequence_length': 10, 'dropout': 0.2}],
    }
    run_dir = tmp_path / 'sasrec-test-hnv'
    with pytest.raises(FileNotFoundError, match='GPU smoke'):
        train_dev(data, config, run_dir)
    run_dir.mkdir()
    (run_dir / 'gpu_smoke-hnv.json').write_text('{"passed": false}', encoding='utf-8')
    with pytest.raises(ValueError, match='smoke manifest'):
        train_dev(data, config, run_dir)
