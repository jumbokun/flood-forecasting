#!/usr/bin/env python3
"""Verify every file hash embedded in one or more Phase-1 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', nargs='+', type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def file_records(value: object) -> Iterator[dict]:
    if isinstance(value, dict):
        if {'resolved_path', 'size_bytes', 'sha256'} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from file_records(child)


def main() -> None:
    args = parse_args()
    total = 0
    for manifest_path in args.manifest:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        count = 0
        for record in file_records(payload):
            path = Path(record['resolved_path'])
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != record['size_bytes']:
                raise RuntimeError(f'size mismatch: {path}')
            if sha256(path) != record['sha256']:
                raise RuntimeError(f'hash mismatch: {path}')
            count += 1
        if not count:
            raise RuntimeError(f'no file records found in {manifest_path}')
        total += count
        print(f'{manifest_path}: verified {count} files')
    print(f'verified {total} file records')


if __name__ == '__main__':
    main()
