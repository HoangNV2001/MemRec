"""Label-independent and budget-matched directed Stage-R evidence."""

from src.memory.graph import UserItemGraph
from src.memory.packer import SnippetPacker
from src.memory.transition_pruner import TransitionEvidencePruner
from src.utils import load_config


class TinyDataset:
    name = 'instructrec-books'
    train_data = {
        0: [1, 2, 3],
        1: [1, 4, 7],
        2: [2, 4, 8],
        3: [3, 5, 9],
    }
    valid_data = {0: 42}
    test_data = {0: 99}
    item_metadata = {
        item: {'title': f'Book {item}', 'description': f'Description {item}'}
        for item in [1, 2, 3, 4, 5, 7, 8, 9, 42, 99]
    }


def test_train_only_directed_evidence_and_label_independence():
    dataset = TinyDataset()
    graph = UserItemGraph(dataset)
    method = TransitionEvidencePruner(dataset, k=8, mix_min_users=1, mix_min_items=2)
    left = method.prune(0, graph, candidates=[99, 42])
    right = method.prune(0, graph, candidates=[42, 99])
    assert left == right
    assert method.graph_stats['temporal_batch_pairs'] == 8
    assert len(left['neighbors']) <= 8
    assert left['n_users'] >= 1
    assert left['n_items'] >= 2
    transition_nodes = [node for node in left['neighbors'] if 'transition_from' in node]
    assert transition_nodes
    assert all(node['id'] not in {1, 2, 3, 42, 99} for node in transition_nodes)
    assert all(node['transition_from'] in {1, 2, 3} for node in transition_nodes)
    assert len({(node['type'], node['id']) for node in left['neighbors']}) == len(left['neighbors'])
    snippet = SnippetPacker().build_neighbor_snippet(transition_nodes[0], dataset)
    assert 'Observed next after Item-' in snippet


def test_no_new_successor_returns_exact_baseline():
    dataset = TinyDataset()
    graph = UserItemGraph(dataset)
    method = TransitionEvidencePruner(dataset, k=8, mix_min_users=1, mix_min_items=2)
    dataset.train_data[0] = [99]
    assert method.prune(0, graph, candidates=[42]) == method.baseline.prune(0, graph, [42])


def test_method_config_differs_only_in_pruner_mode():
    baseline = load_config('configs/memrec_instructrec-books_dev700.yaml')
    method = load_config('configs/memrec_instructrec-books_dev700_transition_stage_r.yaml')
    assert method['memrec']['pruner']['mode'] == 'transition_one_step'
    method['memrec']['pruner']['mode'] = 'llm_rules'
    assert method == baseline
