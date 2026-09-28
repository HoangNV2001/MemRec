"""Offline gates for the single graph-walk intervention in full MemRec."""

from types import SimpleNamespace

import pytest

from src.memory.graph import UserItemGraph
from src.memory.graph_walk_pruner import GraphWalk3Pruner
from src.utils import load_config


def graph():
    # The held-out candidate/target does not enter this frozen training graph.
    data = SimpleNamespace(train_data={
        0: [10, 20],
        1: [10, 30],
        2: [20, 40],
        3: [10, 30],
    })
    return UserItemGraph(data)


def test_walk_reaches_three_hop_items_without_using_candidates():
    frozen = graph()
    pruner = GraphWalk3Pruner(k=4, mix_min_users=1, mix_min_items=3)
    first = pruner.prune(0, frozen, candidates=[30, 999])
    second = pruner.prune(0, frozen, candidates=[40, 998])
    assert first == second
    assert len(first['neighbors']) == 4
    assert len({(node['type'], node['id']) for node in first['neighbors']}) == 4
    assert first['n_items'] >= 3 and first['n_users'] >= 1
    assert any(node['features']['walk_step'] == 3 for node in first['neighbors'])
    assert all(node['score'] > 0 for node in first['neighbors'])
    assert all(node['id'] != 999 for node in first['neighbors'])


def test_walk_handles_isolated_user_and_rejects_oversubscribed_budget():
    pruner = GraphWalk3Pruner(k=4, mix_min_users=1, mix_min_items=2)
    result = pruner.prune(999, graph(), candidates=[10])
    assert result['neighbors'] == []
    with pytest.raises(ValueError):
        GraphWalk3Pruner(k=2, mix_min_users=2, mix_min_items=1)


def test_graph_config_changes_only_pruner_mode(monkeypatch):
    for name in ('MEMREC_SELFHOST_MODEL', 'MEMREC_SELFHOST_REVISION',
                 'MEMREC_SELFHOST_BASE_URL', 'MEMREC_SELFHOST_API_KEY'):
        monkeypatch.setenv(name, 'test-value')
    baseline = load_config('configs/memrec_instructrec-books_dev700.yaml')
    method = load_config('configs/memrec_instructrec-books_dev700_graph_walk3.yaml')
    assert method['memrec']['pruner']['mode'] == 'graph_walk3'
    method['memrec']['pruner']['mode'] = 'llm_rules'
    assert method == baseline
