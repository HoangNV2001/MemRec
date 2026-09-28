"""Temporal direction and frozen-rank fusion checks without LLM calls."""

from types import SimpleNamespace

from scripts import evaluate_books_memrec_transition as transfer


def test_transition_graph_uses_only_ordered_train_history():
    dataset = SimpleNamespace(train_data={0: [1, 2, 3], 1: [1, 4]},
                              valid_data={0: 8, 1: 9}, test_data={0: 10, 1: 11})
    graph, stats = transfer.build_graph(dataset)
    assert graph == {'1': [('2',), ('4',)], '2': [('3',)]}
    assert stats['temporal_batch_pairs'] == 3
    assert all(str(item) not in graph for item in (8, 9, 10, 11))


def test_transition_score_does_not_read_target_and_keeps_failure_miss(monkeypatch):
    monkeypatch.setattr(transfer, 'MONTE_CARLO_WALKS', 100)
    dataset = SimpleNamespace(train_data={0: [1, 2, 3], 1: [1, 4], 2: [2, 5]},
                              valid_data={0: 8, 1: 9, 2: 7})
    graph, _ = transfer.build_graph(dataset)
    row = {'user_id': 0, 'target_item': 4,
           'candidates': [4, 5, 6, 10, 11, 12, 13, 14, 15, 16],
           'ranked_items': [5, 4, 6, 10, 11, 12, 13, 14, 15, 16],
           'target_position': 1, 'failure': None}
    scored = transfer.score_row(row, dataset, graph)
    changed_label = {**row, 'target_item': 5}
    assert scored == transfer.score_row(changed_label, dataset, graph)
    transfer.validate_score_row(scored, row)
    failed_source = {**row, 'target_position': 10, 'failure': 'malformed_ranking'}
    failed = transfer.score_row(failed_source, dataset, graph)
    result = transfer.summarize([failed], [failed_source])
    assert result['predictions'][0]['target_positions'] == {
        'memrec': 10, 'one_step': 10, 'ppr': 10}
