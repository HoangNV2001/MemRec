#!/usr/bin/env python3
"""Download the preregistered Books LLM checkpoint under MemRec-owned storage."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path('/mnt/data/users/anhnct/memrec-hnv')
MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'
REVISION = '5a5a776300a41aaa681dd7ff0106608ef2bc90db'
DESTINATION = ROOT / 'models/Qwen3-30B-A3B-Instruct-2507-FP8-hnv'
MARKER = DESTINATION / 'download-complete-hnv.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Model download must run inside an authorized Slurm step')
    if not str(Path(os.environ.get('HF_HOME', '')).resolve()).startswith(str(ROOT / 'cache')):
        raise RuntimeError('HF_HOME must be under the MemRec-owned cache')
    if shutil.disk_usage(ROOT).free < 80 * 2**30:
        raise RuntimeError('Less than 80 GiB free on /mnt/data; refusing model download')
    if MARKER.exists():
        manifest = json.loads(MARKER.read_text())
        if manifest.get('revision') == REVISION:
            print(json.dumps({'status': 'already_downloaded', **manifest}, sort_keys=True))
            return
        raise RuntimeError('Existing model marker has a different revision')

    snapshot_download(repo_id=MODEL, revision=REVISION,
                      local_dir=DESTINATION, max_workers=2)
    weights = sorted(DESTINATION.glob('*.safetensors'))
    total_bytes = sum(path.stat().st_size for path in weights)
    if not weights or total_bytes < 25_000_000_000:
        raise RuntimeError(f'Incomplete checkpoint: {len(weights)} shards, {total_bytes} bytes')
    small_files = ('config.json', 'tokenizer_config.json')
    for name in small_files:
        if not (DESTINATION / name).is_file():
            raise RuntimeError(f'Missing checkpoint file: {name}')
    manifest = {
        'model': MODEL, 'revision': REVISION,
        'weight_shards': len(weights), 'weight_bytes': total_bytes,
        'small_file_sha256': {name: sha256(DESTINATION / name) for name in small_files},
    }
    MARKER.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'downloaded', **manifest}, sort_keys=True))


if __name__ == '__main__':
    main()
