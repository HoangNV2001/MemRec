"""
Frozen Stage-R environment.

The RL environment is a *snapshot*: the memory graph is materialised once
(RL_PLAN.md §2, static environment) and never written to during training. This
module turns a live ``MemRecAgent`` into that snapshot, and reads it back.

Two properties are load-bearing and worth stating explicitly, because they are
what make the snapshot cheap and exactly faithful:

* ``LLMRulePruner.prune()`` ignores its ``candidates`` argument, so N'_k(u) is
  already candidate-blind and depends only on the frozen graph. Caching it changes
  nothing about the pipeline's behaviour.
* ``SnippetPacker`` renders the neighbour table from *static item metadata*, not
  from the evolving M_v. So the only memory-dependent input to Stage-R is the
  target user's own M_u. Item memories still matter downstream -- Stage-ReRank
  shows them for the candidates -- so we snapshot those too.

The neighbour table is produced by calling the repo's own packer, not a
reimplementation, so the RL prompt's neighbour section is byte-identical to what
the original Stage-R sees (minus the candidate block, §5.3).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import json

SNAPSHOT_VERSION = 1


@dataclass
class UserState:
    """Everything Stage-R needs for one user, with the graph frozen."""

    user_id: int
    user_memory: str
    neighbors_text: str
    neighbor_ids: List[str]
    n_train_items: int

    def to_dict(self) -> Dict:
        return {
            "user_memory": self.user_memory,
            "neighbors_text": self.neighbors_text,
            "neighbor_ids": self.neighbor_ids,
            "n_train_items": self.n_train_items,
        }

    @classmethod
    def from_dict(cls, user_id: int, d: Dict) -> "UserState":
        return cls(
            user_id=int(user_id),
            user_memory=d.get("user_memory", ""),
            neighbors_text=d.get("neighbors_text", ""),
            neighbor_ids=list(d.get("neighbor_ids", [])),
            n_train_items=int(d.get("n_train_items", 0)),
        )


class GraphSnapshot:
    """Serialisable frozen memory graph for the Stage-R environment."""

    def __init__(self, meta: Optional[Dict] = None):
        self.meta: Dict = meta or {}
        self.users: Dict[int, UserState] = {}
        self.item_memories: Dict[int, str] = {}

    # -- access -------------------------------------------------------------

    def state(self, user_id: int) -> UserState:
        return self.users[int(user_id)]

    def item_memory(self, item_id: int) -> str:
        return self.item_memories.get(int(item_id), "")

    def __len__(self) -> int:
        return len(self.users)

    # -- io -----------------------------------------------------------------

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SNAPSHOT_VERSION,
            "meta": self.meta,
            "users": {str(uid): st.to_dict() for uid, st in sorted(self.users.items())},
            "item_memories": {str(iid): m for iid, m in sorted(self.item_memories.items())},
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "GraphSnapshot":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        version = payload.get("version")
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"snapshot version {version} != expected {SNAPSHOT_VERSION} ({path})"
            )
        snap = cls(meta=payload.get("meta", {}))
        snap.users = {
            int(uid): UserState.from_dict(uid, d) for uid, d in payload.get("users", {}).items()
        }
        snap.item_memories = {
            int(iid): m for iid, m in payload.get("item_memories", {}).items()
        }
        return snap


def neighbor_id_strings(pruned_subgraph: Dict) -> List[str]:
    """``[{'type': 'item', 'id': 5}, ...]`` -> ``['Item-5', ...]`` (repo convention)."""
    out = []
    for nb in pruned_subgraph.get("neighbors", []):
        out.append(f"{str(nb['type']).capitalize()}-{int(nb['id'])}")
    return out


# SnippetPacker.pack() reserves 300 tokens for the candidate block whenever
# candidates are passed. Dropping candidates would silently hand the policy a
# *larger* neighbour budget than the prompted Stage-R baseline gets (1000 vs 700
# tokens), which would confound "GRPO beats prompted" with "saw more neighbours".
# We shrink tau by the same 300 so the neighbour table is budgeted identically and
# the only difference from the original Stage-R input is the missing candidates.
CANDIDATE_BLOCK_RESERVE = 300


def build_user_state(agent, user_id: int) -> UserState:
    """
    Materialise one user's frozen state using the agent's own pruner and packer.

    ``candidates=None`` everywhere: this is the candidate-blind state of §5.3.
    """
    from src.memory import SnippetPacker

    pruned = agent.pruner.prune(user_id, agent.graph, None)
    user_memory_summary = agent.storage.render_user_summary(user_id)
    packer = SnippetPacker(tau=agent.packer.tau - CANDIDATE_BLOCK_RESERVE)
    packed = packer.pack(
        pruned_subgraph=pruned,
        dataset=agent.dataset,
        candidates=None,
        user_memory_summary=user_memory_summary,
    )
    raw_memory = agent.storage.get_user_memory(user_id) or ""
    return UserState(
        user_id=int(user_id),
        user_memory=raw_memory,
        neighbors_text=packed["neighbors_text"],
        neighbor_ids=neighbor_id_strings(pruned),
        n_train_items=len(agent.dataset.get_user_train_items(user_id)),
    )


def build_snapshot(
    agent,
    user_ids: Sequence[int],
    extra_item_ids: Iterable[int] = (),
    meta: Optional[Dict] = None,
) -> GraphSnapshot:
    """
    Freeze the graph for ``user_ids``.

    ``extra_item_ids`` should carry every item that will ever appear in a
    candidate list, so the reward ranker can read its memory without loading the
    600MB metadata file.
    """
    snap = GraphSnapshot(meta=meta)

    wanted_items = {int(i) for i in extra_item_ids}
    for user_id in user_ids:
        state = build_user_state(agent, int(user_id))
        snap.users[state.user_id] = state
        for nid in state.neighbor_ids:
            kind, _, num = nid.partition("-")
            if kind == "Item":
                wanted_items.add(int(num))

    for item_id in sorted(wanted_items):
        memory = agent.storage.get_item_memory(item_id)
        if memory:
            snap.item_memories[int(item_id)] = memory

    return snap
