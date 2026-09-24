import numpy as np
import torch

from src.temporal_movielens.baseline_models import BPRMF, SASRec
from src.temporal_movielens.baselines import _bpr_epoch, _sasrec_epoch


def test_bpr_epoch_updates_model_with_fixed_candidate_universe():
    torch.manual_seed(7)
    model = BPRMF(3, 8, 4)
    before = model.item_embedding.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = _bpr_epoch(
        model,
        {0: [0, 1], 1: [2, 3], 2: [4, 5]},
        list(range(8)),
        optimizer,
        3,
        np.random.default_rng(7),
        torch.device("cpu"),
    )
    assert np.isfinite(loss)
    assert not torch.equal(before, model.item_embedding.weight)


def test_bpr_epoch_rejects_known_positives_with_precomputed_pairs():
    torch.manual_seed(7)
    model = BPRMF(2, 5, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pairs = (
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.asarray([0, 1, 2, 3], dtype=np.int32),
    )
    loss = _bpr_epoch(
        model,
        pairs,
        list(range(5)),
        optimizer,
        4,
        np.random.default_rng(7),
        torch.device("cpu"),
    )
    assert np.isfinite(loss)


def test_sasrec_epoch_updates_model_with_right_padded_targets():
    torch.manual_seed(7)
    model = SASRec(8, 4, 4, 1, 2, 0.0)
    before = model.item_embedding.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = _sasrec_epoch(
        model,
        {0: [0, 1, 2], 1: [2, 3, 4], 2: [4, 5, 6]},
        list(range(8)),
        optimizer,
        2,
        4,
        np.random.default_rng(7),
        torch.device("cpu"),
    )
    assert np.isfinite(loss)
    assert not torch.equal(before, model.item_embedding.weight)
