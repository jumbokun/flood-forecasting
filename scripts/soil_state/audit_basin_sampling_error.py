#!/usr/bin/env python3
"""Quantify 256-cell basin sampling error against all contained DWD cells."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from shapely import contains_xy


DEPTHS = ('0-10', '0-30')
QUANTILES = (10, 50, 90)


def dwd_name(year: int, depth: str) -> str:
    return f'grids_germany_daily_soil_moisture_composite_{year}_{depth}_v1.nc'


def full_cell_indices(geometry, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    minx, miny, maxx, maxy = geometry.bounds
    cols = np.flatnonzero((x >= minx) & (x <= maxx))
    rows = np.flatnonzero((y >= miny) & (y <= maxy))
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    inside = contains_xy(
        geometry,
        x[col_grid.ravel()],
        y[row_grid.ravel()],
    )
    return row_grid.ravel()[inside] * len(x) + col_grid.ravel()[inside]


def load_selected_grids(
    path: Path, target_dates: list[np.datetime64]
) -> dict[np.datetime64, np.ndarray]:
    with Dataset(path) as dataset:
        paws = dataset.variables['paws']
        time = dataset.variables['time']
        dates = num2date(
            time[:],
            units=time.units,
            calendar=getattr(time, 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
        )
        date_values = np.array(
            [np.datetime64(item.date(), 'D') for item in dates]
        )
        lookup = {value: index for index, value in enumerate(date_values)}
        paws.set_auto_maskandscale(False)
        raw = np.asarray(paws[:])
        if raw.ndim == 4 and raw.shape[0] == 1:
            raw = raw[0]
        fill = int(getattr(paws, '_FillValue', -9999))
        scale = float(getattr(paws, 'scale_factor', 1.0))
        result = {}
        for target in target_dates:
            values = raw[lookup[target]].reshape(-1).astype(np.float32)
            values[values == fill] = np.nan
            result[target] = values * np.float32(scale)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-root', required=True, type=Path)
    parser.add_argument('--boundaries', required=True, type=Path)
    parser.add_argument('--daily-root', required=True, type=Path)
    parser.add_argument('--output-csv', required=True, type=Path)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--basin-sample-count', type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling = pd.read_csv(args.daily_root / 'basin_sampling.csv')
    subsampled = sampling[
        sampling['contained_cell_count'] > sampling['selected_cell_count']
    ]
    if len(subsampled) < args.basin_sample_count:
        raise RuntimeError('not enough subsampled basins')
    rank_indices = np.linspace(
        0, len(subsampled) - 1, args.basin_sample_count, dtype=int
    )
    chosen = (
        subsampled.sort_values('contained_cell_count').iloc[rank_indices].copy()
    )

    payload = np.load(
        args.daily_root / 'daily_features.npz', allow_pickle=False
    )
    basin_values = payload['basin'].astype(str)
    basin_lookup = {value: index for index, value in enumerate(basin_values)}
    daily_dates = payload['date'].astype('datetime64[D]')
    date_lookup = {value: index for index, value in enumerate(daily_dates)}
    sample_index_payload = np.load(
        args.daily_root / 'sampling_index.npz', allow_pickle=False
    )
    x = sample_index_payload['grid_x']
    y = sample_index_payload['grid_y']

    boundaries = (
        gpd.read_file(args.boundaries).to_crs(31467).set_index('gauge_id')
    )
    test_dates = [
        np.datetime64(f'{year}-{month:02d}-15', 'D')
        for year in (2021, 2024)
        for month in (1, 3, 5, 7, 9, 11)
    ]
    grids: dict[tuple[str, np.datetime64], np.ndarray] = {}
    for depth in DEPTHS:
        for year in (2021, 2024):
            year_dates = [
                date for date in test_dates if int(str(date)[:4]) == year
            ]
            loaded = load_selected_grids(
                args.archive_root / dwd_name(year, depth), year_dates
            )
            grids.update(
                {(depth, date): values for date, values in loaded.items()}
            )

    rows: list[dict[str, object]] = []
    for record in chosen.itertuples(index=False):
        geometry = boundaries.loc[record.gauge_id].geometry
        indices = full_cell_indices(geometry, x, y)
        if len(indices) != int(record.contained_cell_count):
            raise RuntimeError(f'cell-count drift for {record.gauge_id}')
        basin_idx = basin_lookup[record.basin]
        for depth in DEPTHS:
            slug = depth.replace('-', '_')
            for date in test_dates:
                exact_values = grids[(depth, date)][indices]
                exact_values = exact_values[np.isfinite(exact_values)]
                if not len(exact_values):
                    continue
                exact = np.percentile(exact_values, QUANTILES)
                date_idx = date_lookup[date]
                for quantile, exact_value in zip(QUANTILES, exact, strict=True):
                    sampled_value = float(
                        payload[f'dwd_soil_{slug}_p{quantile}'][
                            basin_idx, date_idx
                        ]
                    )
                    rows.append(
                        {
                            'basin': record.basin,
                            'gauge_id': record.gauge_id,
                            'contained_cell_count': int(
                                record.contained_cell_count
                            ),
                            'selected_cell_count': int(
                                record.selected_cell_count
                            ),
                            'date': str(date),
                            'depth_cm': depth,
                            'quantile': quantile,
                            'exact_percent_nfk': float(exact_value),
                            'sampled_percent_nfk': sampled_value,
                            'error_percent_nfk': sampled_value
                            - float(exact_value),
                            'absolute_error_percent_nfk': abs(
                                sampled_value - float(exact_value)
                            ),
                        }
                    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = np.array([float(row['error_percent_nfk']) for row in rows])
    summary = {
        'schema': 'camels-de-1h-dwd-sampling-error-audit-v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'basin_sample_count': len(chosen),
        'dates': [str(value) for value in test_dates],
        'depths_cm': list(DEPTHS),
        'quantiles': list(QUANTILES),
        'comparison_count': len(rows),
        'bias_percent_nfk': float(errors.mean()),
        'mae_percent_nfk': float(np.abs(errors).mean()),
        'rmse_percent_nfk': float(np.sqrt(np.mean(errors**2))),
        'p95_absolute_error_percent_nfk': float(
            np.percentile(np.abs(errors), 95)
        ),
        'max_absolute_error_percent_nfk': float(np.abs(errors).max()),
        'pass_gate': bool(np.percentile(np.abs(errors), 95) <= 10.0),
        'gate': 'p95 absolute quantile error <= 10 %nFK',
        'output_csv': str(args.output_csv.resolve()),
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
