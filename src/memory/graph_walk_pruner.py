"""Parameter-free three-step graph evidence selection for full MemRec.

The walk uses only permitted training edges. Candidate IDs and evaluation
labels are deliberately not used for selecting evidence; candidates remain
available to the unchanged Stage-R and ReRank prompts.
"""

from collections import defaultdict


class GraphWalk3Pruner:
    """Select the existing k-node context from 2/3-step walk mass."""

    def __init__(self, k: int, mix_min_users: int, mix_min_items: int):
        if k < 1 or mix_min_users < 0 or mix_min_items < 0:
            raise ValueError('Invalid graph-walk neighbor budget')
        if mix_min_users + mix_min_items > k:
            raise ValueError('Graph-walk type minima exceed k')
        self.k = k
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        self._cache = {}
        self._graph = None

    def _walk(self, user_id: int, graph):
        if graph is not self._graph:
            self._cache.clear()
            self._graph = graph
        if user_id in self._cache:
            return self._cache[user_id]

        history = tuple(dict.fromkeys(graph.get_user_items(user_id)))
        if not history:
            self._cache[user_id] = ((), ())
            return self._cache[user_id]
        # P2(v|u): choose a historical item uniformly, then one of its
        # training users uniformly. Remove the starting user, then normalize.
        users = defaultdict(float)
        for item_id in history:
            owners = set(graph.get_item_users(item_id))
            if not owners:
                continue
            mass = 1.0 / (len(history) * len(owners))
            for neighbor_id in owners:
                if neighbor_id != user_id:
                    users[neighbor_id] += mass
        total_user_mass = sum(users.values())
        if total_user_mass:
            for neighbor_id in users:
                users[neighbor_id] /= total_user_mass

        # P3(i|u): a third step from P2 to the neighbor's training items.
        # Direct history still appears when the walk returns to it. Only an
        # isolated user falls back to P1; there are no mixing weights.
        items = defaultdict(float)
        if users:
            for neighbor_id, user_mass in users.items():
                neighbor_items = set(graph.get_user_items(neighbor_id))
                if not neighbor_items:
                    continue
                mass = user_mass / len(neighbor_items)
                for item_id in neighbor_items:
                    items[item_id] += mass
        else:
            for item_id in history:
                items[item_id] = 1.0 / len(history)

        item_rank = tuple(sorted(
            items.items(),
            key=lambda row: (-row[1], -graph.get_item_recency(user_id, row[0]), row[0]),
        )[:self.k])
        user_rank = tuple(sorted(users.items(), key=lambda row: (-row[1], row[0]))[:self.k])
        self._cache[user_id] = (item_rank, user_rank)
        return self._cache[user_id]

    def prune(self, user_id: int, graph, candidates=None):
        item_rank, user_rank = self._walk(user_id, graph)
        history_set = set(graph.get_user_items(user_id))
        item_nodes = [
            {'type': 'item', 'id': item_id, 'score': score,
             'features': {'walk_step': 1 if item_id in history_set else 3}}
            for item_id, score in item_rank
        ]
        user_nodes = [
            {'type': 'user', 'id': neighbor_id, 'score': score,
             'features': {'walk_step': 2}}
            for neighbor_id, score in user_rank
        ]
        selected = item_nodes[:self.mix_min_items] + user_nodes[:self.mix_min_users]
        remaining = item_nodes[self.mix_min_items:] + user_nodes[self.mix_min_users:]
        remaining.sort(key=lambda node: (-node['score'], node['type'], node['id']))
        selected.extend(remaining[:max(0, self.k - len(selected))])
        selected.sort(key=lambda node: (-node['score'], node['type'], node['id']))
        return {
            'user_id': user_id,
            'neighbors': selected,
            'n_items': sum(node['type'] == 'item' for node in selected),
            'n_users': sum(node['type'] == 'user' for node in selected),
            'meta': {'k': self.k, 'mode': 'graph_walk3',
                     'mix_min_users': self.mix_min_users,
                     'mix_min_items': self.mix_min_items},
        }
