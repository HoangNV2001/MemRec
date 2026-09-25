"""Optional exact-input cache for costly self-host MemRec LLM responses.

The cache is a performance/resume aid, never a source of new evidence. The
namespace must pin the model revision and backend contract. SQLite's rollback
journal is used because the project cache resides on shared network storage.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class ExactResponseCache:
    def __init__(self, path: str, namespace: str):
        if not namespace:
            raise ValueError('Exact-response cache requires a nonempty namespace')
        self.namespace = namespace
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(destination), timeout=30, check_same_thread=False)
        self.connection.execute('PRAGMA journal_mode=DELETE')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS responses ('
            'key TEXT PRIMARY KEY, response TEXT NOT NULL, '
            'prompt_tokens INTEGER, completion_tokens INTEGER, has_usage INTEGER NOT NULL)'
        )
        self.connection.commit()
        self.lock = threading.Lock()

    def key(self, request: dict) -> str:
        payload = json.dumps(
            {'namespace': self.namespace, 'request': request},
            sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def get(self, key: str):
        with self.lock:
            row = self.connection.execute(
                'SELECT response, prompt_tokens, completion_tokens, has_usage '
                'FROM responses WHERE key=?', (key,)
            ).fetchone()
        return row

    def put(self, key: str, response: str, prompt_tokens, completion_tokens, has_usage: bool):
        if not isinstance(response, str):
            return
        with self.lock:
            self.connection.execute(
                'INSERT OR IGNORE INTO responses VALUES (?, ?, ?, ?, ?)',
                (key, response, prompt_tokens, completion_tokens, int(has_usage)),
            )
            self.connection.commit()
