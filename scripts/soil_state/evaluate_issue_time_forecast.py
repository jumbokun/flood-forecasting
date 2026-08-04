#!/usr/bin/env python3
"""Evaluate overlapping forecasts on an explicit issue-time x lead-time grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from googlehydrology.datasetzoo.multimet import MultimetDataLoader
from googlehydrology.evaluation import get_tester
from googlehydrology.evaluation.utils import get_samples_indexes
from googlehydrology.utils.config import Config


class IssueTimeBasinBatchSampler:
    """Yield basin-local batches only at the requested issue-time cadence."""

    def __init__(
        self,
        *,
        sample_index,
        basin_indexes: np.ndarray,
        dataset_dates: pd.DatetimeIndex,
        batch_size: int,
        issue_stride_hours: int,
    ):
        basin_column = sample_index.get_column('basin')
        date_column = sample_index.get_column('date')
        hours_from_epoch = (
            dataset_dates - pd.Timestamp('1970-01-01')
        ) // pd.Timedelta(hours=1)
        keep_date = np.asarray(hours_from_epoch) % issue_stride_hours == 0
        self._batches: list[range | list[int]] = []
        for basin_index in basin_indexes:
            start = int(np.searchsorted(basin_column, basin_index, side='left'))
            end = int(np.searchsorted(basin_column, basin_index, side='right'))
            positions = np.arange(start, end, dtype=np.int64)
            positions = positions[keep_date[date_column[start:end]]]
            for offset in range(0, len(positions), batch_size):
                batch = positions[offset : offset + batch_size]
                self._batches.append(batch.tolist())

    def __iter__(self) -> Iterator[list[int] | range]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_pair(
    obs: np.ndarray, sim: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask].astype(np.float64), sim[mask].astype(np.float64)


def calculate_metrics(
    obs: np.ndarray, sim: np.ndarray
) -> dict[str, float | int]:
    obs, sim = finite_pair(obs, sim)
    result: dict[str, float | int] = {'n': int(len(obs))}
    if len(obs) < 5:
        return result

    residual = sim - obs
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    result['nse'] = (
        float(1 - np.sum(residual**2) / denominator)
        if denominator > 0
        else math.nan
    )
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    if obs_std > 0 and obs_mean != 0:
        correlation = (
            float(np.corrcoef(obs, sim)[0, 1]) if sim_std > 0 else math.nan
        )
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean
        result['kge'] = float(
            1
            - np.sqrt(
                (correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
            )
        )
        result['alpha'] = float(alpha)
        result['beta'] = float(beta)
    result['rmse_mm_h'] = float(np.sqrt(np.mean(residual**2)))
    result['mae_mm_h'] = float(np.mean(np.abs(residual)))
    result['mean_bias_mm_h'] = float(np.mean(residual))
    result['volume_bias_fraction'] = (
        float(np.sum(residual) / np.sum(obs)) if np.sum(obs) != 0 else math.nan
    )
    result['negative_prediction_fraction'] = float(np.mean(sim < 0))

    top_count = max(1, int(math.ceil(0.02 * len(obs))))
    top_indices = np.argsort(obs)[-top_count:]
    top_obs = obs[top_indices]
    top_sim = sim[top_indices]
    result['fhv_percent'] = (
        float(100 * np.sum(top_sim - top_obs) / np.sum(top_obs))
        if np.sum(top_obs) != 0
        else math.nan
    )
    threshold = np.quantile(obs, 0.95)
    high = obs >= threshold
    result['q95_threshold_mm_h'] = float(threshold)
    result['q95_underprediction_fraction'] = float(
        np.mean(sim[high] < obs[high])
    )
    result['q95_bias_fraction'] = (
        float(np.sum(sim[high] - obs[high]) / np.sum(obs[high]))
        if np.sum(obs[high]) != 0
        else math.nan
    )
    result['peak_bias_fraction'] = (
        float((np.max(sim) - np.max(obs)) / np.max(obs))
        if np.max(obs) != 0
        else math.nan
    )
    return result


def load_weights(
    model: torch.nn.Module, checkpoint: Path, device: torch.device
) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model_keys = set(model.state_dict())
    state_keys = set(state)
    model_prefixed = any(key.startswith('_orig_mod.') for key in model_keys)
    state_prefixed = any(key.startswith('_orig_mod.') for key in state_keys)
    if model_prefixed and not state_prefixed:
        state = {f'_orig_mod.{key}': value for key, value in state.items()}
    elif state_prefixed and not model_prefixed:
        state = {
            key.removeprefix('_orig_mod.'): value
            for key, value in state.items()
        }
    model.load_state_dict(state)


def summarize(frame: pd.DataFrame) -> dict:
    metric_columns = [
        column
        for column in frame.columns
        if column not in {'basin', 'lead_hour', 'n'}
    ]
    by_lead = {}
    for lead, group in frame.groupby('lead_hour', sort=True):
        by_lead[str(int(lead))] = {
            f'median_{column}': (
                float(group[column].median(skipna=True))
                if group[column].notna().any()
                else None
            )
            for column in metric_columns
        } | {'n_basins': int(group['basin'].nunique())}

    bands = {
        'nowcast_0h': (0, 0),
        'forecast_1_6h': (1, 6),
        'forecast_7_12h': (7, 12),
        'forecast_13_24h': (13, 24),
        'forecast_25_48h': (25, 48),
        'forecast_49_72h_no_nwp_tail': (49, 72),
    }
    by_band = {}
    for name, (start, end) in bands.items():
        group = frame[frame['lead_hour'].between(start, end, inclusive='both')]
        if group.empty:
            continue
        by_band[name] = {
            f'median_{column}': (
                float(group[column].median(skipna=True))
                if group[column].notna().any()
                else None
            )
            for column in metric_columns
        } | {
            'lead_hours': [start, end],
            'n_basin_lead_rows': int(len(group)),
        }
    return {'by_lead': by_lead, 'by_lead_band': by_band}


def forcing_support_contract(frame: pd.DataFrame) -> dict:
    """Label forecast leads by the forcing actually available at issue time."""

    maximum = int(frame['lead_hour'].max())
    result = {
        'observed_nowcast_lead_hours': [0, 0],
        'native_nwp_lead_hours': [1, min(48, maximum)],
    }
    if maximum > 48:
        result['hydrologic_tail_without_nwp_lead_hours'] = [
            49,
            maximum,
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument(
        '--period', choices=['validation', 'test'], default='test'
    )
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--issue-stride-hours', type=int, default=3)
    parser.add_argument('--basin-file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    config_path = args.run_dir / 'config.yml'
    cfg = Config(config_path)
    cfg.update_config(
        {
            'batch_size': args.batch_size,
            # Per-basin evaluation creates several batch shapes. Compiling each
            # shape triggers repeated Triton autotuning and is substantially
            # slower than eager inference for this workload.
            'compile': False,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'inference_mode': True,
        }
    )
    if args.basin_file is not None:
        cfg.update_config(
            {
                f'{args.period}_basin_file': args.basin_file,
            }
        )
    checkpoints = sorted(args.run_dir.glob('model_epoch*.pt'))
    checkpoint = (
        args.run_dir / f'model_epoch{args.epoch:03d}.pt'
        if args.epoch is not None
        else checkpoints[-1]
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    epoch = int(checkpoint.stem.removeprefix('model_epoch'))
    output_dir = args.output_dir or (
        args.run_dir / 'issue_time_evaluation' / args.period / checkpoint.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tester = get_tester(
        cfg=cfg,
        run_dir=args.run_dir,
        period=args.period,
        init_model=True,
    )
    load_weights(tester.model, checkpoint, tester.device)
    # We call the low-level generator directly; unlike Tester.evaluate(), it
    # does not switch off dropout for us.
    tester.model.eval()
    basins = list(tester.basins)
    batch_sampler = IssueTimeBasinBatchSampler(
        sample_index=tester.dataset._sample_index,
        batch_size=args.batch_size,
        basin_indexes=get_samples_indexes(tester.basins, samples=basins),
        dataset_dates=pd.DatetimeIndex(tester.dataset._dataset['date'].values),
        issue_stride_hours=args.issue_stride_hours,
    )
    loader = MultimetDataLoader(
        tester.dataset,
        lazy_load=cfg.lazy_load,
        logging_level=cfg.logging_level,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=tester.dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    target = cfg.target_variables[0]
    scaler = tester.dataset.scaler.scaler[target]
    center = float(scaler.sel(parameter='center'))
    scale = float(scaler.sel(parameter='scale'))
    frequency = tester.dataset.frequencies[0]
    rows = []
    issue_counts = {}
    started = time.time()
    evaluation = tester._evaluate(
        tester.model, loader, tester.dataset.frequencies, set(basins)
    )
    for basin_number, basin_data in enumerate(evaluation, start=1):
        basin = basin_data['basin']
        obs = basin_data['obs'][frequency].numpy()[:, :, 0]
        sim = basin_data['preds'][frequency].numpy()[:, :, 0]
        dates = basin_data['dates'][frequency]
        obs = obs * scale + center
        sim = sim * scale + center

        issue_times = pd.DatetimeIndex(dates[:, 0])
        if args.issue_stride_hours > 1:
            hours_from_epoch = (
                issue_times - pd.Timestamp('1970-01-01')
            ) // pd.Timedelta(hours=1)
            keep = np.asarray(hours_from_epoch) % args.issue_stride_hours == 0
            issue_times = issue_times[keep]
            dates = dates[keep]
            obs = obs[keep]
            sim = sim[keep]
        if issue_times.has_duplicates:
            raise ValueError(f'Duplicate issue times remain for {basin}.')
        lead_hours = ((dates[0] - dates[0, 0]) / np.timedelta64(1, 'h')).astype(
            int
        )
        expected_dates = issue_times.values[:, None] + lead_hours[
            None, :
        ].astype('timedelta64[h]')
        if not np.array_equal(dates, expected_dates):
            raise ValueError(
                f'Valid-time grid does not equal issue_time + lead for {basin}.'
            )
        issue_counts[basin] = int(len(issue_times))
        for column, lead in enumerate(lead_hours):
            rows.append(
                {
                    'basin': basin,
                    'lead_hour': int(lead),
                    **calculate_metrics(obs[:, column], sim[:, column]),
                }
            )
        if basin_number % 10 == 0 or basin_number == len(basins):
            pd.DataFrame(rows).to_csv(
                output_dir / 'metrics_by_basin_lead.csv', index=False
            )
            print(
                f'{basin_number}/{len(basins)} basins; '
                f'elapsed={time.time() - started:.1f}s',
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary = {
        'schema_version': 1,
        'description': (
            'Issue-time-honest forecast evaluation. Rows are distinct '
            'forecast issue times; columns span the configured lead '
            f'hours {int(frame["lead_hour"].min())}..'
            f'{int(frame["lead_hour"].max())}.'
        ),
        'run_dir': str(args.run_dir),
        'period': args.period,
        'epoch': epoch,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'config_sha256': sha256(config_path),
        'script_sha256': sha256(Path(__file__)),
        'target': target,
        'target_unit': 'mm h-1',
        'issue_stride_hours': args.issue_stride_hours,
        'n_basins': int(frame['basin'].nunique()),
        'n_basin_lead_rows': int(len(frame)),
        'issue_count_min': int(min(issue_counts.values())),
        'issue_count_median': float(np.median(list(issue_counts.values()))),
        'issue_count_max': int(max(issue_counts.values())),
        'elapsed_seconds': float(time.time() - started),
        'forcing_support_contract': forcing_support_contract(frame),
        **summarize(frame),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    print(json.dumps(summary['by_lead_band'], indent=2), flush=True)


if __name__ == '__main__':
    main()
