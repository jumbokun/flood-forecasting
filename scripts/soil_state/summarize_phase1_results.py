#!/usr/bin/env python3
"""Summarize matched Phase-1 issue-time soil-state experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = {
    'nowcast_0h': (0, 0),
    'forecast_1_6h': (1, 6),
    'forecast_7_12h': (7, 12),
    'forecast_13_24h': (13, 24),
    'forecast_25_48h': (25, 48),
}
METRICS = {
    'nse': 'higher',
    'kge': 'higher',
    'rmse_mm_h': 'lower',
    'mae_mm_h': 'lower',
    'abs_fhv_percent': 'lower',
    'q95_underprediction_fraction': 'lower',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument(
        '--evaluation-subdir', default='issue_time_evaluation_daily'
    )
    parser.add_argument(
        '--stage', choices=('screen', 'confirmation'), default='screen'
    )
    parser.add_argument('--checkpoint-epoch', type=int)
    parser.add_argument('--bootstrap-replicates', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260804)
    return parser.parse_args()


def experiment_parts(name: str) -> tuple[str, int]:
    fields = name.split('_')
    return fields[2], int(fields[-1].removeprefix('s'))


def read_results(
    manifest: dict, evaluation_subdir: str, checkpoint_epoch: int
) -> pd.DataFrame:
    frames = []
    for experiment, record in manifest['runs'].items():
        variant, seed = experiment_parts(experiment)
        path = (
            Path(record['run_dir'])
            / evaluation_subdir
            / f'test/model_epoch{checkpoint_epoch:03d}/metrics_by_basin_lead.csv'
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if len(frame) != 297 * 49:
            raise RuntimeError(f'Unexpected row count for {path}: {len(frame)}')
        if frame[['basin', 'lead_hour']].duplicated().any():
            raise RuntimeError(f'Duplicate basin/lead rows in {path}')
        frame['variant'] = variant
        frame['seed'] = seed
        frame['experiment'] = experiment
        frame['abs_fhv_percent'] = frame['fhv_percent'].abs()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def band_rows(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return frame[frame['lead_hour'].between(start, end, inclusive='both')]


def seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, seed), group in frame.groupby(['variant', 'seed'], sort=True):
        for band, (start, end) in BANDS.items():
            selected = band_rows(group, start, end)
            row = {
                'variant': variant,
                'seed': int(seed),
                'band': band,
                'lead_start': start,
                'lead_end': end,
                'n_basins': int(selected['basin'].nunique()),
                'n_basin_lead_rows': int(len(selected)),
            }
            for metric in METRICS:
                row[f'median_{metric}'] = float(selected[metric].median())
            rows.append(row)
    return pd.DataFrame(rows)


def paired_delta(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
    metric: str,
    direction: str,
) -> pd.Series:
    left = treatment.groupby('basin', sort=True)[metric].median()
    right = control.groupby('basin', sort=True)[metric].median()
    if (
        len(left) != 297
        or len(right) != 297
        or not left.index.equals(right.index)
    ):
        raise RuntimeError(
            f'Expected the same 297 basin keys for {metric}, found '
            f'S1={len(left)} and control={len(right)}'
        )
    joined = pd.concat(
        [left.rename('s1'), right.rename('control')], axis=1
    ).dropna()
    if len(joined) < 200:
        raise RuntimeError(
            f'Too few finite paired basins for {metric}: {len(joined)}'
        )
    if direction == 'higher':
        return joined['s1'] - joined['control']
    return joined['control'] - joined['s1']


def bootstrap_median(
    values: np.ndarray, *, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    medians = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[index] = np.median(sample)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def paired_summary(
    frame: pd.DataFrame, *, replicates: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for seed in sorted(frame['seed'].unique()):
        s1 = frame[(frame['variant'] == 's1') & (frame['seed'] == seed)]
        for control_name in ('c0', 'c1'):
            control = frame[
                (frame['variant'] == control_name) & (frame['seed'] == seed)
            ]
            for band, (start, end) in BANDS.items():
                treatment_band = band_rows(s1, start, end)
                control_band = band_rows(control, start, end)
                for metric, direction in METRICS.items():
                    deltas = paired_delta(
                        treatment_band, control_band, metric, direction
                    )
                    low, high = bootstrap_median(
                        deltas.to_numpy(), replicates=replicates, rng=rng
                    )
                    rows.append(
                        {
                            'seed': seed,
                            'comparison': f's1_minus_{control_name}',
                            'band': band,
                            'metric': metric,
                            'direction_normalized': 'positive_is_s1_better',
                            'paired_median_delta': float(deltas.median()),
                            'paired_mean_delta': float(deltas.mean()),
                            'bootstrap_median_ci_low': low,
                            'bootstrap_median_ci_high': high,
                            'fraction_basins_s1_better': float(
                                (deltas > 0).mean()
                            ),
                            'n_basins': int(len(deltas)),
                        }
                    )
    return pd.DataFrame(rows)


def screen_gate(paired: pd.DataFrame) -> dict:
    selected = paired[
        (paired['band'] == 'forecast_25_48h') & (paired['metric'] == 'nse')
    ]
    comparisons = {}
    for comparison, group in selected.groupby('comparison', sort=True):
        positive = int((group['paired_median_delta'] > 0).sum())
        ci_positive = int((group['bootstrap_median_ci_low'] > 0).sum())
        comparisons[comparison] = {
            'positive_seed_count': positive,
            'ci_excludes_zero_seed_count': ci_positive,
            'median_of_seed_deltas': float(
                group['paired_median_delta'].median()
            ),
        }
    ranking_signal = all(
        result['positive_seed_count'] >= 2 for result in comparisons.values()
    )
    return {
        'status': 'PROMOTE_TO_CONFIRMATION'
        if ranking_signal
        else 'DO_NOT_PROMOTE',
        'rule': (
            'Promote S1 only if its paired median 25-48 h NSE delta is positive '
            'against both C0 and C1 in at least two of three matched seeds.'
        ),
        'comparisons': comparisons,
        'caveat': (
            'This is a deliberately undertrained ranking screen. It cannot by itself '
            'establish production skill or a final no-go conclusion.'
        ),
    }


def confirmation_gate(paired: pd.DataFrame) -> dict:
    selected = paired[
        (paired['band'] == 'forecast_25_48h') & (paired['metric'] == 'nse')
    ]
    comparisons = {}
    for comparison, group in selected.groupby('comparison', sort=True):
        if len(group) != 1:
            raise RuntimeError(
                f'Confirmation expects one matched seed for {comparison}, found {len(group)}'
            )
        row = group.iloc[0]
        comparisons[comparison] = {
            'paired_median_delta': float(row['paired_median_delta']),
            'bootstrap_median_ci_low': float(row['bootstrap_median_ci_low']),
            'bootstrap_median_ci_high': float(row['bootstrap_median_ci_high']),
            'fraction_basins_s1_better': float(
                row['fraction_basins_s1_better']
            ),
            'n_finite_paired_basins': int(row['n_basins']),
        }
    positive = all(
        result['paired_median_delta'] > 0
        and result['bootstrap_median_ci_low'] > 0
        for result in comparisons.values()
    )
    return {
        'status': 'GO_TO_PHASE2' if positive else 'NO_GO_OR_INCONCLUSIVE',
        'rule': (
            "Advance only if S1's paired median 25-48 h NSE delta is positive "
            'and its basin-bootstrap 95% interval excludes zero against both C0 '
            'and the longer-history C1 control.'
        ),
        'comparisons': comparisons,
        'caveat': (
            'A pass authorizes full multi-seed training and ablation, not production '
            'deployment or a direct conversion from DWD soil moisture to Curve Number.'
        ),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.run_manifest.read_text(encoding='utf-8'))
    checkpoint_epoch = args.checkpoint_epoch
    if checkpoint_epoch is None:
        checkpoint_epoch = 3 if args.stage == 'screen' else 5
    frame = read_results(manifest, args.evaluation_subdir, checkpoint_epoch)
    rng = np.random.default_rng(args.bootstrap_seed)
    seeds = seed_summary(frame)
    paired = paired_summary(
        frame, replicates=args.bootstrap_replicates, rng=rng
    )
    gate = (
        screen_gate(paired)
        if args.stage == 'screen'
        else confirmation_gate(paired)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'screen' if args.stage == 'screen' else 'confirmation'
    seeds.to_csv(args.output_dir / f'{prefix}_seed_summary.csv', index=False)
    paired.to_csv(
        args.output_dir / f'{prefix}_paired_bootstrap.csv', index=False
    )
    payload = {
        'schema_version': 1,
        'evaluation_subdir': args.evaluation_subdir,
        'bootstrap_replicates': args.bootstrap_replicates,
        'bootstrap_seed': args.bootstrap_seed,
        'stage': args.stage,
        'checkpoint_epoch': checkpoint_epoch,
        'n_experiments': int(frame['experiment'].nunique()),
        'n_test_basins': int(frame['basin'].nunique()),
        'lead_hours': [
            int(frame['lead_hour'].min()),
            int(frame['lead_hour'].max()),
        ],
        f'{prefix}_gate': gate,
    }
    (args.output_dir / f'{prefix}_conclusion.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
