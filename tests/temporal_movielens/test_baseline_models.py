import pytest
import torch

from src.temporal_movielens.baseline_models import BPRMF, SASRec, right_padded_sequences


def test_bpr_pairwise_and_candidate_scores_are_finite():
    torch.manual_seed(7)
    model = BPRMF(n_users=3, n_items=5, embedding_dim=4)
    users = torch.tensor([0, 1])
    positive = torch.tensor([1, 2])
    negative = torch.tensor([3, 4])
    loss = model.pairwise_loss(users, positive, negative)
    scores = model.score_candidates(users, torch.tensor([[1, 3], [2, 4]]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert scores.shape == (2, 2) and torch.isfinite(scores).all()


def test_sasrec_training_and_candidate_scoring_shapes():
    torch.manual_seed(7)
    model = SASRec(
        n_items=8,
        embedding_dim=4,
        maximum_sequence_length=4,
        transformer_blocks=1,
        attention_heads=2,
        dropout=0.0,
    )
    inputs = torch.tensor([[1, 2, 3, 0], [2, 4, 0, 0]])
    positives = torch.tensor([[2, 3, 4, 0], [4, 5, 0, 0]])
    negatives = torch.tensor([[6, 7, 8, 0], [7, 8, 0, 0]])
    loss = model.pointwise_loss(inputs, positives, negatives)
    scores = model.score_candidates(
        inputs,
        torch.tensor([3, 2]),
        torch.tensor([[1, 4], [2, 5]]),
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert scores.shape == (2, 2) and torch.isfinite(scores).all()


def test_right_padded_sequences_preserve_recent_zero_based_items():
    values, lengths = right_padded_sequences([[0, 2, 4, 6], [3]], 3)
    assert values.tolist() == [[3, 5, 7], [4, 0, 0]]
    assert lengths.tolist() == [3, 1]
    with pytest.raises(ValueError):
        right_padded_sequences([[]], 3)
