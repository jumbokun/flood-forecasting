#!/usr/bin/env python3
"""Freeze exact Phase-1 soil-state experiment inputs and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


RUNTIME_FILES = [
    'googlehydrology/datasetzoo/multimet.py',
    'googlehydrology/modelzoo/mean_embedding_forecast_lstm.py',
    'googlehydrology/training/basetrainer.py',
    'googlehydrology/utils/config.py',
    'googlehydrology/datautils/scaler.py',
    'googlehydrology/evaluation/tester.py',
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ['git', '-C', str(repo), *args], text=True
    ).strip()


def file_record(path: Path) -> dict:
    resolved = path.resolve()
    return {
        'path': str(path),
        'resolved_path': str(resolved),
        'size_bytes': resolved.stat().st_size,
        'sha256': sha256(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--runtime-repo', required=True, type=Path)
    parser.add_argument('--source-worktree', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-audit', action='append', default=[], type=Path)
    parser.add_argument(
        '--stage', choices=('screen', 'confirm'), default='screen'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == 'screen':
        seeds = (111, 222, 333)
        epochs = 3
        updates_per_epoch = 500
        checkpoint_epoch = 3
    else:
        seeds = (111,)
        epochs = 5
        updates_per_epoch = 2000
        checkpoint_epoch = 5
    experiments = [
        f'soil_state_{variant}_{args.stage}_s{seed}'
        for variant in ('c0', 'c1', 's1')
        for seed in seeds
    ]
    runs = {}
    for experiment in experiments:
        matches = sorted(args.run_root.glob(f'{experiment}_*'))
        if len(matches) != 1:
            raise RuntimeError(
                f'Expected exactly one run for {experiment}, found {len(matches)}: {matches}'
            )
        run_dir = matches[0].resolve()
        files = {
            name: file_record(run_dir / name)
            for name in (
                'config.yml',
                f'model_epoch{checkpoint_epoch:03d}.pt',
                'scaler.nc',
                'output.log',
            )
        }
        evaluation_root = (
            run_dir
            / 'issue_time_evaluation_daily'
            / 'test'
            / f'model_epoch{checkpoint_epoch:03d}'
        )
        files['evaluation_metrics'] = file_record(
            evaluation_root / 'metrics_by_basin_lead.csv'
        )
        files['evaluation_summary'] = file_record(
            evaluation_root / 'summary.json'
        )
        runs[experiment] = {'run_dir': str(run_dir), 'files': files}

    runtime_files = {}
    for relative in RUNTIME_FILES:
        path = args.runtime_repo / relative
        runtime_files[relative] = file_record(path)

    source_files = {}
    for path in sorted(
        (args.source_worktree / 'configs/soil_state_phase1').glob('*.yml')
    ):
        source_files[str(path.relative_to(args.source_worktree))] = file_record(
            path
        )
    for path in sorted(
        (args.source_worktree / 'scripts/soil_state').glob('*.py')
    ):
        source_files[str(path.relative_to(args.source_worktree))] = file_record(
            path
        )

    payload = {
        'schema_version': 1,
        'generated_at_utc': datetime.now(UTC).isoformat(),
        'experiment_contract': {
            'stage': args.stage,
            'variants': {
                'c0': '720 h meteorological hindcast',
                'c1': '1440 h meteorological hindcast (DWD-free history control)',
                's1': '720 h meteorological hindcast plus causal DWD soil-state features',
            },
            'seeds': list(seeds),
            'epochs': epochs,
            'updates_per_epoch': updates_per_epoch,
            'checkpoint_epoch': checkpoint_epoch,
        },
        'runs': runs,
        'data_audits': [file_record(path) for path in args.data_audit],
        'runtime': {
            'repo': str(args.runtime_repo.resolve()),
            'git_head': git(args.runtime_repo, 'rev-parse', 'HEAD'),
            'git_status_porcelain': git(
                args.runtime_repo, 'status', '--porcelain'
            ),
            'files': runtime_files,
        },
        'source_worktree': {
            'path': str(args.source_worktree.resolve()),
            'git_head': git(args.source_worktree, 'rev-parse', 'HEAD'),
            'git_branch': git(args.source_worktree, 'branch', '--show-current'),
            'git_status_porcelain': git(
                args.source_worktree, 'status', '--porcelain'
            ),
            'files': source_files,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    print(f'froze {len(runs)} runs to {args.output}')


if __name__ == '__main__':
    main()
