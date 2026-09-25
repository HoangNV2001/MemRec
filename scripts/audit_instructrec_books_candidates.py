#!/usr/bin/env python3
"""Read-only audit of original InstructRec Books test candidate lists.

Run the 30-user smoke first, then explicitly opt in to the full audit:
    python scripts/audit_instructrec_books_candidates.py
    python scripts/audit_instructrec_books_candidates.py --full
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/instructrec-books'


def audit(limit: int | None) -> dict:
    source = pd.read_pickle(DATA / 'booksAll_recagent.pkl')
    interactions = pd.read_csv(DATA / 'instructrec-books.inter', sep='\t')
    sequences = interactions.sort_values(['user_id', 'timestamp']).groupby('user_id').item_id.agg(list)
    metadata_ids = set(pd.read_csv(DATA / 'instructrec-books.meta', sep='\t', usecols=['item_id']).item_id)

    count = len(source) if limit is None else min(limit, len(source))
    digest = hashlib.sha256()
    positions = Counter()
    for user_id in range(count):
        row = source.iloc[user_id]
        history = [int(item) for item in row['asin']]
        candidates = [int(item) for item in row['ranked_lists']]
        if history != sequences.loc[user_id]:
            raise ValueError(f'User {user_id}: converted sequence differs from source')
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError(f'User {user_id}: invalid candidate count or duplicates')
        if history[-1] not in candidates:
            raise ValueError(f'User {user_id}: target missing from candidates')
        if set(candidates).intersection(history[:-1]):
            raise ValueError(f'User {user_id}: candidates include earlier interaction')
        if not set(candidates).issubset(metadata_ids):
            raise ValueError(f'User {user_id}: candidate metadata missing')
        positions[candidates.index(history[-1]) + 1] += 1
        digest.update(json.dumps([user_id, history[-1], candidates], separators=(',', ':')).encode())
        digest.update(b'\n')

    return {
        'users_audited': count,
        'source_users': len(source),
        'candidate_count_per_user': 10,
        'target_position_counts': dict(sorted(positions.items())),
        'ordered_candidates_sha256': digest.hexdigest(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Audit all users after the 30-user smoke')
    args = parser.parse_args()
    print(json.dumps(audit(None if args.full else 30), indent=2))
