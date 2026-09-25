#!/usr/bin/env python3
"""Matched SASRec baseline: CPU smoke, GPU smoke, or locked dev training."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.books_sasrec import cpu_smoke, gpu_smoke, load_books_data, train_dev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--cpu-smoke', action='store_true')
    action.add_argument('--gpu-smoke', action='store_true')
    action.add_argument('--train-dev', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    config_path = ROOT / 'configs/books_sasrec_baseline.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    data = load_books_data()
    if args.cpu_smoke:
        result = cpu_smoke(data, config)
    else:
        if args.output_dir is None:
            parser.error('GPU actions require --output-dir under the MemRec run root')
        result = gpu_smoke(data, config, args.output_dir) if args.gpu_smoke else train_dev(data, config, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
