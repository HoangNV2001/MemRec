#!/usr/bin/env python3
"""Block full Books GPU work unless the promoted real-LLM smoke is intact."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.check_books_memrec_smoke import check as check_smoke


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_ast(source: str, kind: str) -> str:
    tree = ast.parse(source)
    if kind == 'client':
        names = {'validate_json_shape', '_strip_provider_prefix', '_restricted_sampling_params'}
        selected = [node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        selected.extend(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'LLMClient')
    elif kind == 'warmup':
        selected = [method for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'MemRecTrainer'
                    for method in node.body if isinstance(method, ast.FunctionDef)
                    and method.name == '_warmup_single_user']
    else:
        raise ValueError(kind)
    if not selected:
        raise ValueError(f'Missing semantic AST section: {kind}')
    selected_module = ast.Module(body=selected, type_ignores=[])
    return hashlib.sha256(ast.dump(selected_module, include_attributes=False).encode()).hexdigest()


def verify(smoke_dir: Path, repo: Path, revision: str, cache_namespace: str) -> dict:
    promotion = json.loads((smoke_dir / 'promotion.json').read_text())
    if promotion['status'] != 'passed' or promotion['model_revision'] != revision:
        raise ValueError('Unpromoted or wrong-revision smoke')
    if promotion['cache_namespace'] != cache_namespace:
        raise ValueError('LLM response cache namespace differs from smoke')
    for name, expected in promotion['artifact_sha256'].items():
        if digest(smoke_dir / name) != expected:
            raise ValueError(f'Smoke artifact changed: {name}')
    if check_smoke(smoke_dir)['status'] != 'pass':
        raise ValueError('Smoke output gate no longer passes')

    smoke_commit = promotion['git_commit']
    commit_exists = subprocess.run(
        ['git', '-C', str(repo), 'cat-file', '-e', f'{smoke_commit}^{{commit}}'],
        check=False, capture_output=True,
    ).returncode == 0
    if not commit_exists:
        raise ValueError('Smoke source commit is not available locally')
    for source_name, kind in (
        ('src/models/llm_client.py', 'client'),
        ('src/train/trainer_memrec.py', 'warmup'),
    ):
        old = subprocess.check_output(
            ['git', '-C', str(repo), 'show', f'{smoke_commit}:{source_name}'], text=True
        )
        current = (repo / source_name).read_text()
        if semantic_ast(old, kind) != semantic_ast(current, kind):
            raise ValueError(f'Inference semantics changed since smoke: {source_name}')

    for line in (smoke_dir / 'source-hashes.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name in {'src/models/llm_client.py', 'src/train/trainer_memrec.py'}:
            # Only resumability was added; the inference/warm-up ASTs are
            # compared to the promoted smoke commit above.
            continue
        path = Path(name)
        if not path.is_absolute():
            path = repo / path
        if digest(path) != expected:
            raise ValueError(f'Smoke model/config/prompt file changed: {name}')
    return {'status': 'pass', 'smoke_run_id': promotion['run_id'],
            'smoke_commit': smoke_commit, 'model_revision': revision}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-dir', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--cache-namespace', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.smoke_dir, args.repo, args.revision,
                            args.cache_namespace), sort_keys=True))


if __name__ == '__main__':
    main()
