#!/usr/bin/env python3
"""Aggregate DWD 1 km daily composite soil state to CAMELS-DE-1h basins.

The aggregation uses a deterministic sample of at most N DWD cell centres per
basin. This keeps nested national catchments tractable while preserving an
auditable spatial distribution. Small catchments without a contained grid-cell
centre use the cell nearest their representative point and are flagged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from shapely import contains_xy


DEPTHS = ('0-10', '0-30')
QUANTILES = (10, 50, 90)


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = map(int, part.split('-', 1))
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    return sorted(years)


def load_model_basins(basins_root: Path) -> list[str]:
    basin_ids: set[str] = set()
    for split in ('train', 'val', 'test'):
        path = basins_root / f'{split}_basins.txt'
        if not path.is_file():
            raise FileNotFoundError(path)
        for raw in path.read_text(encoding='utf-8').splitlines():
            basin = raw.strip()
            if basin:
                basin_ids.add(basin)
    return sorted(basin_ids)


def dwd_name(year: int, depth: str) -> str:
    return f'grids_germany_daily_soil_moisture_composite_{year}_{depth}_v1.nc'


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def nearest_grid_index(
    x: np.ndarray, y: np.ndarray, px: float, py: float
) -> int:
    col = int(np.argmin(np.abs(x - px)))
    row = int(np.argmin(np.abs(y - py)))
    return row * len(x) + col


def build_sampling_index(
    boundaries: Path,
    model_basins: list[str],
    x: np.ndarray,
    y: np.ndarray,
    max_cells: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    frame = gpd.read_file(boundaries).to_crs(31467)
    if 'gauge_id' not in frame:
        raise RuntimeError('catchment boundaries have no gauge_id')
    frame = frame.set_index('gauge_id', drop=False)
    gauge_ids = [basin.removeprefix('camelsde1h_') for basin in model_basins]
    missing = sorted(set(gauge_ids).difference(frame.index))
    if missing:
        raise RuntimeError(
            f'{len(missing)} model basins lack boundaries: {missing[:10]}'
        )

    padded = np.full((len(model_basins), max_cells), -1, dtype=np.int32)
    selected_counts = np.zeros(len(model_basins), dtype=np.int16)
    audit: list[dict[str, Any]] = []
    for basin_index, (model_basin, gauge_id) in enumerate(
        zip(model_basins, gauge_ids, strict=True)
    ):
        geometry = frame.loc[gauge_id].geometry
        minx, miny, maxx, maxy = geometry.bounds
        cols = np.flatnonzero((x >= minx) & (x <= maxx))
        rows = np.flatnonzero((y >= miny) & (y <= maxy))
        fallback = False
        if len(cols) and len(rows):
            row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
            candidate_x = x[col_grid.ravel()]
            candidate_y = y[row_grid.ravel()]
            inside = contains_xy(geometry, candidate_x, candidate_y)
            indices = (
                row_grid.ravel()[inside] * len(x) + col_grid.ravel()[inside]
            )
        else:
            indices = np.empty(0, dtype=np.int64)
        full_count = int(len(indices))
        if full_count == 0:
            point = geometry.representative_point()
            indices = np.array(
                [nearest_grid_index(x, y, point.x, point.y)], dtype=np.int64
            )
            fallback = True
        if len(indices) > max_cells:
            rng = np.random.default_rng(stable_seed(gauge_id))
            indices = np.sort(
                rng.choice(indices, size=max_cells, replace=False)
            )
        count = len(indices)
        padded[basin_index, :count] = indices.astype(np.int32)
        selected_counts[basin_index] = count
        audit.append(
            {
                'basin': model_basin,
                'gauge_id': gauge_id,
                'area_calc_km2': float(
                    frame.loc[gauge_id].get('area_calc', np.nan)
                ),
                'contained_cell_count': full_count,
                'selected_cell_count': count,
                'selection_fraction': min(1.0, count / full_count)
                if full_count
                else 0.0,
                'nearest_cell_fallback': fallback,
            }
        )
    return padded, selected_counts, audit


def read_grid_contract(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path) as dataset:
        x = np.asarray(dataset.variables['x'][:], dtype=np.float64)
        y = np.asarray(dataset.variables['y'][:], dtype=np.float64)
        if x.shape != (654,) or y.shape != (866,):
            raise RuntimeError(
                f'unexpected DWD grid in {path}: {x.shape}, {y.shape}'
            )
        if not np.allclose(np.diff(x), 1000.0) or not np.allclose(
            np.diff(y), -1000.0
        ):
            raise RuntimeError(f'unexpected DWD grid spacing in {path}')
        return x, y


def aggregate_file(
    path: Path,
    expected_year: int,
    sample_indices: np.ndarray,
    selected_counts: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    safe_indices = np.where(sample_indices >= 0, sample_indices, 0)
    padded = sample_indices < 0
    with Dataset(path) as dataset:
        paws = dataset.variables['paws']
        time = dataset.variables['time']
        dates = num2date(
            time[:],
            units=time.units,
            calendar=getattr(time, 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
        )
        if dates[0].year != expected_year or dates[-1].year != expected_year:
            raise RuntimeError(f'date/year mismatch in {path}')
        paws.set_auto_maskandscale(False)
        raw = np.asarray(paws[:])
        if raw.ndim == 4 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 3 or raw.shape[1:] != (866, 654):
            raise RuntimeError(f'unexpected paws shape in {path}: {paws.shape}')
        date_array = np.array(
            [np.datetime64(item.date(), 'D') for item in dates]
        )
        order = np.argsort(date_array, kind='stable')
        date_array = date_array[order]
        raw = raw[order]
        if len(np.unique(date_array)) != len(date_array):
            raise RuntimeError(f'duplicate time coordinates in {path}')
        expected_dates = np.arange(
            np.datetime64(f'{expected_year}-01-01'),
            np.datetime64(f'{expected_year + 1}-01-01'),
            dtype='datetime64[D]',
        )
        if not np.array_equal(date_array, expected_dates):
            raise RuntimeError(f'incomplete daily time coordinate in {path}')
        fill = int(getattr(paws, '_FillValue', -9999))
        scale = float(getattr(paws, 'scale_factor', 1.0))
        samples_raw = raw.reshape(len(dates), -1)[:, safe_indices]
        valid = (samples_raw != fill) & ~padded[None, :, :]
        samples = samples_raw.astype(np.float32) * np.float32(scale)
        samples[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            quantiles = np.nanpercentile(samples, QUANTILES, axis=2).astype(
                np.float32
            )
        coverage = valid.sum(axis=2, dtype=np.int32) / selected_counts[None, :]
        result = {
            'p10': quantiles[0].T,
            'p50': quantiles[1].T,
            'p90': quantiles[2].T,
            'coverage': coverage.T.astype(np.float32),
        }
        return date_array, result


def write_sampling_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-root', required=True, type=Path)
    parser.add_argument('--boundaries', required=True, type=Path)
    parser.add_argument('--basins-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--years', default='2019-2024')
    parser.add_argument('--max-cells', type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    if not years or args.max_cells < 1:
        raise ValueError('years and max-cells must be non-empty/positive')
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_root.iterdir()}
    resumable_staging = {'sampling_index.npz', 'basin_sampling.csv'}
    if existing.difference(resumable_staging):
        raise RuntimeError(
            f'output-root contains non-staging output and cannot be resumed: '
            f'{sorted(existing)}'
        )

    reference = args.archive_root / dwd_name(years[0], DEPTHS[0])
    x, y = read_grid_contract(reference)
    basins = load_model_basins(args.basins_root)
    sample_indices, selected_counts, sampling_audit = build_sampling_index(
        args.boundaries, basins, x, y, args.max_cells
    )
    np.savez_compressed(
        output_root / 'sampling_index.npz',
        basin=np.asarray(basins),
        flat_cell_index=sample_indices,
        selected_cell_count=selected_counts,
        grid_x=x,
        grid_y=y,
    )
    write_sampling_csv(output_root / 'basin_sampling.csv', sampling_audit)

    by_depth: dict[str, dict[str, list[np.ndarray]]] = {
        depth: {name: [] for name in ('p10', 'p50', 'p90', 'coverage')}
        for depth in DEPTHS
    }
    expected_dates: np.ndarray | None = None
    for depth in DEPTHS:
        depth_dates: list[np.ndarray] = []
        for year in years:
            path = args.archive_root / dwd_name(year, depth)
            dates, features = aggregate_file(
                path, year, sample_indices, selected_counts
            )
            depth_dates.append(dates)
            for name, values in features.items():
                by_depth[depth][name].append(values)
            print(
                json.dumps(
                    {
                        'year': year,
                        'depth_cm': depth,
                        'days': len(dates),
                        'finite_p50_fraction': float(
                            np.isfinite(features['p50']).mean()
                        ),
                        'mean_coverage': float(
                            np.nanmean(features['coverage'])
                        ),
                    }
                ),
                flush=True,
            )
        dates_all = np.concatenate(depth_dates)
        if expected_dates is None:
            expected_dates = dates_all
        elif not np.array_equal(expected_dates, dates_all):
            raise RuntimeError('depth date axes differ')

    assert expected_dates is not None
    arrays: dict[str, np.ndarray] = {
        'basin': np.asarray(basins),
        'date': expected_dates,
    }
    for depth in DEPTHS:
        slug = depth.replace('-', '_')
        for name, parts in by_depth[depth].items():
            arrays[f'dwd_soil_{slug}_{name}'] = np.concatenate(parts, axis=1)
    arrays['dwd_soil_layer_gradient_p50'] = (
        arrays['dwd_soil_0_30_p50'] - arrays['dwd_soil_0_10_p50']
    ).astype(np.float32)
    np.savez_compressed(output_root / 'daily_features.npz', **arrays)

    audit = {
        'schema': 'camels-de-1h-dwd-basin-daily-v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_archive': str(args.archive_root.resolve()),
        'source_boundaries': str(args.boundaries.resolve()),
        'source_basins': str(args.basins_root.resolve()),
        'years': years,
        'depths_cm': list(DEPTHS),
        'quantiles': list(QUANTILES),
        'basin_count': len(basins),
        'date_count': len(expected_dates),
        'max_cells_per_basin': args.max_cells,
        'nearest_cell_fallback_count': sum(
            bool(row['nearest_cell_fallback']) for row in sampling_audit
        ),
        'subsampled_basin_count': sum(
            int(row['contained_cell_count']) > args.max_cells
            for row in sampling_audit
        ),
        'mean_selected_cell_count': float(selected_counts.mean()),
        'features': sorted(
            key for key in arrays if key not in ('basin', 'date')
        ),
        'daily_features_bytes': (output_root / 'daily_features.npz')
        .stat()
        .st_size,
        'causality_note': (
            'These are valid-date aggregates, not issue-time features. The hourly '
            'materializer applies an explicit conservative availability lag.'
        ),
    }
    (output_root / 'build_audit.json').write_text(
        json.dumps(audit, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
