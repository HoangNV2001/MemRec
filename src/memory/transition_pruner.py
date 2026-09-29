"""Budget-matched one-step temporal evidence for MemRec's Stage-R.

The directed graph is built exclusively from consecutive train interactions.
It changes the collaborative evidence read by Stage-R, not the final ranking.
"""

from src.temporal_common.graph import build_transition_graph_from_users, one_step_scores

from .pruner_llm_rules import LLMRulePruner


class TransitionEvidencePruner:
    """Replace non-mandatory baseline neighbors with directed successors."""

    def __init__(self, dataset, k=16, mix_min_users=4, mix_min_items=6, seed_items=6):
        self.dataset = dataset
        self.k = k
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        self.seed_items = seed_items
        self.baseline = LLMRulePruner(
            dataset_name=getattr(dataset, 'name', 'instructrec-books'),
            k=k, dataset=dataset,
            mix_min_users=mix_min_users, mix_min_items=mix_min_items,
        )
        histories = (
            [(position + 1, str(item_id)) for position, item_id in enumerate(items)]
            for items in dataset.train_data.values()
        )
        self.transitions, self.graph_stats = build_transition_graph_from_users(
            histories, cutoff=10**9,
        )
        expected = sum(max(len(items) - 1, 0) for items in dataset.train_data.values())
        if self.graph_stats['temporal_batch_pairs'] != expected:
            raise ValueError('Temporal graph contains non-train transitions')

    def prune(self, user_id, graph, candidates=None):
        baseline = self.baseline.prune(user_id, graph, candidates)
        selected = baseline['neighbors']
        history = self.dataset.train_data.get(user_id, [])
        seeds = list(dict.fromkeys(str(item) for item in reversed(history)))[:self.seed_items]
        if not seeds:
            return baseline

        # Candidate IDs and evaluation labels are deliberately not used here.
        successors = {
            int(item)
            for source in seeds
            for destinations in self.transitions.get(source, ())
            for item in destinations
        }
        history_ids = set(history)
        selected_ids = {(node['type'], node['id']) for node in selected}
        metadata = self.dataset.item_metadata or {}
        eligible = sorted(
            item for item in successors
            if item not in history_ids
            and ('item', item) not in selected_ids
            and metadata.get(item, {}).get('title')
        )
        if not eligible:
            return baseline
        probabilities = one_step_scores(seeds, [str(item) for item in eligible], self.transitions)
        eligible.sort(key=lambda item: (-probabilities[str(item)], item))

        # Retain the original mix minima; only the unconstrained slots can be
        # replaced. The historical six recent seeds and k/mix budgets are frozen.
        users = [node for node in selected if node['type'] == 'user']
        items = [node for node in selected if node['type'] == 'item']
        protected = users[:self.mix_min_users] + items[:self.mix_min_items]
        protected_ids = {(node['type'], node['id']) for node in protected}
        remaining = [node for node in selected if (node['type'], node['id']) not in protected_ids]
        n_transition = min(len(eligible), self.k - len(protected))
        if n_transition <= 0:
            return baseline
        base_priority = max((node['score'] for node in selected), default=0.0)
        additions = []
        for item in eligible[:n_transition]:
            source = next(
                int(seed) for seed in seeds
                if any(str(item) in destinations for destinations in self.transitions.get(seed, ()))
            )
            additions.append({
                'type': 'item', 'id': item, 'score': probabilities[str(item)],
                'pack_priority': base_priority + 1 + probabilities[str(item)],
                'transition_from': source,
                'transition_probability': probabilities[str(item)],
                'features': {'directed_one_step': True},
            })
        augmented = protected + additions + remaining[:self.k - len(protected) - len(additions)]
        if len(augmented) > self.k or len({(n['type'], n['id']) for n in augmented}) != len(augmented):
            raise ValueError('Temporal evidence violated the neighbor budget')
        return {
            'user_id': user_id, 'neighbors': augmented,
            'n_items': sum(node['type'] == 'item' for node in augmented),
            'n_users': sum(node['type'] == 'user' for node in augmented),
        }
