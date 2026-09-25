"""Transactional per-user progress for the long serial Books full-MemRec run."""

import hashlib
import json
import sqlite3
from pathlib import Path


def contract_digest(contract: dict) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class BooksRunJournal:
    def __init__(self, path: str, contract: dict):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(destination), timeout=30)
        self.connection.execute('PRAGMA journal_mode=DELETE')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS metadata ('
            'id INTEGER PRIMARY KEY CHECK(id=1), contract_sha256 TEXT NOT NULL)'
        )
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS entries ('
            'phase TEXT NOT NULL, seq INTEGER NOT NULL, user_id INTEGER NOT NULL, '
            'payload TEXT NOT NULL, PRIMARY KEY (phase, seq))'
        )
        digest = contract_digest(contract)
        self.connection.execute('INSERT OR IGNORE INTO metadata VALUES (1, ?)', (digest,))
        self.connection.commit()
        stored = self.connection.execute(
            'SELECT contract_sha256 FROM metadata WHERE id=1'
        ).fetchone()[0]
        if stored != digest:
            raise ValueError('Books run journal contract differs from the resumed run')
        self.contract_sha256 = digest

    def read_phase(self, phase: str, expected_user_ids: list[int]) -> list[dict]:
        rows = self.connection.execute(
            'SELECT seq, user_id, payload FROM entries WHERE phase=? ORDER BY seq',
            (phase,),
        ).fetchall()
        if len(rows) > len(expected_user_ids):
            raise ValueError(f'Journal has too many {phase} entries')
        records = []
        for expected_seq, (seq, user_id, payload) in enumerate(rows):
            if seq != expected_seq or user_id != expected_user_ids[expected_seq]:
                raise ValueError(f'Non-contiguous or wrong-user {phase} journal at {expected_seq}')
            records.append(json.loads(payload))
        return records

    def append(self, phase: str, seq: int, user_id: int, payload: dict) -> None:
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            previous = self.connection.execute(
                'SELECT MAX(seq) FROM entries WHERE phase=?', (phase,)
            ).fetchone()[0]
            next_seq = 0 if previous is None else previous + 1
            if seq != next_seq:
                raise ValueError(f'Expected {phase} journal seq {next_seq}, got {seq}')
            self.connection.execute(
                'INSERT INTO entries VALUES (?, ?, ?, ?)',
                (phase, seq, user_id,
                 json.dumps(payload, separators=(',', ':'), ensure_ascii=False)),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
