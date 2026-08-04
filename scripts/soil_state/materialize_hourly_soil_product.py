#!/usr/bin/env python3
"""Materialize causal hourly DWD basin features and a composed model-data root."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def causal_source_indices(
    daily_dates: np.ndarray,
    issue_dates: np.ndarray,
    availability_lag_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    cutoff = issue_dates - np.timedelta64(availability_lag_hours, 'h')
    cutoff_days = cutoff.astype('datetime64[D]')
    indices = np.searchsorted(daily_dates, cutoff_days, side='right') - 1
    available = indices >= 0
    safe = np.maximum(indices, 0)
    return safe, available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily-npz', required=True, type=Path)
    parser.add_argument('--soil-product-root', required=True, type=Path)
    parser.add_argument('--source-model-data-root', required=True, type=Path)
    parser.add_argument('--composed-model-data-root', required=True, type=Path)
    parser.add_argument('--hourly-start', default='2020-01-01T00:00:00')
    parser.add_argument('--hourly-end', default='2024-12-31T23:00:00')
    parser.add_argument('--availability-lag-hours', type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.availability_lag_hours < 24:
        raise ValueError('historical replay lag below 24 h is not accepted')
    product_root = args.soil_product_root.resolve()
    zarr_path = product_root / 'timeseries.zarr'
    if product_root.exists() and any(product_root.iterdir()):
        raise RuntimeError(f'soil-product-root must be empty: {product_root}')
    product_root.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.daily_npz, allow_pickle=False)
    basins = payload['basin'].astype(str)
    daily_dates = payload['date'].astype('datetime64[D]')
    issue_dates = pd.date_range(
        args.hourly_start, args.hourly_end, freq='1h'
    ).values
    source_indices, available = causal_source_indices(
        daily_dates, issue_dates, args.availability_lag_hours
    )
    source_dates = daily_dates[source_indices]
    age_hours = (
        (issue_dates - source_dates.astype('datetime64[ns]'))
        / np.timedelta64(1, 'h')
    ).astype(np.float32)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for name in payload.files:
        if name in ('basin', 'date'):
            continue
        daily_values = payload[name].astype(np.float32, copy=False)
        hourly_values = daily_values[:, source_indices]
        hourly_values[:, ~available] = np.nan
        data_vars[name] = (('basin', 'date'), hourly_values)
    age = np.broadcast_to(
        age_hours[None, :], (len(basins), len(issue_dates))
    ).copy()
    age[:, ~available] = np.nan
    data_vars['dwd_soil_state_age_hours'] = (('basin', 'date'), age)
    availability = np.broadcast_to(
        available.astype(np.float32)[None, :], (len(basins), len(issue_dates))
    ).copy()
    data_vars['dwd_soil_temporally_available'] = (
        ('basin', 'date'),
        availability,
    )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={'basin': basins, 'date': issue_dates},
        attrs={
            'schema': 'camels-de-1h-dwd-soil-causal-hourly-v1',
            'availability_lag_hours': args.availability_lag_hours,
            'availability_contract': (
                'Feature valid date d is exposed only when issue_time >= d + lag; '
                '48 h is a conservative historical replay assumption because exact '
                'historical DWD publication timestamps are not archived.'
            ),
            'source_daily_npz': str(args.daily_npz.resolve()),
        },
    ).chunk({'basin': 64, 'date': 8760})
    dataset.to_zarr(zarr_path, mode='w', consolidated=True)

    composed = args.composed_model_data_root.resolve()
    if composed.exists() and any(composed.iterdir()):
        raise RuntimeError(
            f'composed-model-data-root must be empty: {composed}'
        )
    composed.mkdir(parents=True, exist_ok=True)
    source = args.source_model_data_root.resolve()
    for item in source.iterdir():
        os.symlink(
            item, composed / item.name, target_is_directory=item.is_dir()
        )
    os.symlink(product_root, composed / 'DWD_SOIL', target_is_directory=True)
    # MultiMet resolves a variable such as ``dwd_soil_0_10_p50`` from the
    # first prefix token and therefore opens product directory ``DWD``.
    os.symlink(product_root, composed / 'DWD', target_is_directory=True)
    audit = {
        'schema': 'camels-de-1h-iconfc-soil-composed-v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_model_data_root': str(source),
        'soil_product_root': str(product_root),
        'composed_model_data_root': str(composed),
        'loader_product_alias': str(composed / 'DWD'),
        'basin_count': len(basins),
        'hourly_date_count': len(issue_dates),
        'hourly_start': str(issue_dates[0]),
        'hourly_end': str(issue_dates[-1]),
        'availability_lag_hours': args.availability_lag_hours,
        'state_age_hours_min': float(np.nanmin(age)),
        'state_age_hours_max': float(np.nanmax(age)),
        'variables': sorted(data_vars),
        'physical_payload_note': 'Only DWD_SOIL is new; DWD and all other dataset members are symlinks.',
    }
    (product_root / 'build_audit.json').write_text(
        json.dumps(audit, indent=2) + '\n', encoding='utf-8'
    )
    (composed / 'soil_composition_audit.json').write_text(
        json.dumps(audit, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
