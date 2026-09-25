"""Immutable user cohorts for the full-MemRec InstructRec Books benchmark.

Only evaluation labels are user-disjoint. Every arm may use the same permitted
training histories to construct collaborative memory and train SASRec.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Collection, Iterable


COHORT_SEED = 'full-memrec-books-v1-20260925'
DEV_SIZE = 2000
PRIOR_1K = Path(__file__).resolve().parents[2] / 'data/eval_user_samples/eval_user_sample_1k_instructrec-books.json'


def cohort_digest(user_ids: Iterable[int]) -> str:
    """Hash sorted IDs in a cross-platform, compact JSON representation."""
    payload = json.dumps(sorted(int(uid) for uid in user_ids), separators=(',', ':'))
    return hashlib.sha256(payload.encode('ascii')).hexdigest()


def split_cohorts(
    user_ids: Collection[int],
    previously_exposed: Collection[int],
    dev_size: int = DEV_SIZE,
    seed: str = COHORT_SEED,
) -> dict[str, list[int]]:
    """Keep every historically inspected user in dev; hash-split the rest."""
    universe = set(int(uid) for uid in user_ids)
    exposed = set(int(uid) for uid in previously_exposed)
    if len(universe) != len(user_ids):
        raise ValueError('Duplicate user IDs in cohort universe')
    if not exposed.issubset(universe):
        raise ValueError('Previously exposed user is absent from the universe')
    if not len(exposed) <= dev_size < len(universe):
        raise ValueError('Development size must include exposed users and leave a held-out set')

    unseen = universe - exposed
    hashed = sorted(
        unseen,
        key=lambda uid: (hashlib.sha256(f'{seed}:{uid}'.encode('ascii')).digest(), uid),
    )
    dev = sorted(exposed | set(hashed[:dev_size - len(exposed)]))
    heldout = sorted(universe - set(dev))
    return {'dev': dev, 'heldout': heldout, 'all': sorted(universe)}


def books_cohorts(user_ids: Collection[int]) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Build Books cohorts, excluding all users scored in historical MemRec runs."""
    ids = sorted(int(uid) for uid in user_ids)
    if ids != list(range(7377)):
        raise ValueError('Books protocol expects original contiguous user IDs 0..7376')
    with PRIOR_1K.open() as handle:
        prior_1k = set(json.load(handle)['user_ids'])
    # Historical 3-user and 100-user runs used random.sample with seed=42 over
    # insertion-ordered 0..7376 user IDs. Five-user runs used the fixed 1k list.
    exposed = set(prior_1k)
    exposed.update(random.Random(42).sample(ids, 3))
    exposed.update(random.Random(42).sample(ids, 100))
    cohorts = split_cohorts(ids, exposed)
    manifest = {
        'seed': COHORT_SEED,
        'dev_size': DEV_SIZE,
        'historically_exposed_users': len(exposed),
        'historically_exposed_sha256': cohort_digest(exposed),
        'cohort_sizes': {name: len(members) for name, members in cohorts.items()},
        'cohort_sha256': {name: cohort_digest(members) for name, members in cohorts.items()},
    }
    return cohorts, manifest
