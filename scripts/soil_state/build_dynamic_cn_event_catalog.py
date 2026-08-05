#!/usr/bin/env python3
"""Build an audited CAMELS-DE-1h rainfall/direct-runoff event catalogue.

The catalogue is designed for the preregistered Germany dynamic-CN
experiment.  Curve Number is never treated as observed ground truth: event
CN values are optional diagnostics inverted from observed rainfall and
baseflow-separated runoff.  The double holdout is written separately so it
is not accidentally opened during model development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


RAIN_COLUMN = 'precipitation_mean_gapfilled'
FLOW_COLUMN = 'discharge_spec_obs'
TEMPERATURE_COLUMN = 'air_temperature_mean'
TIME_COLUMN = 'date'
MODEL_PREFIX = 'camelsde1h_'
EVENT_COLUMNS = [
    'event_id',
    'basin',
    'gauge_id',
    'event_start',
    'rain_end',
    'response_end',
    'season',
    'rain_hours',
    'response_hours',
    'event_rain_mm',
    'rain_coverage',
    'streamflow_coverage',
    'antecedent_rain_5d_mm',
    'antecedent_rain_5d_coverage',
    'antecedent_rain_30d_mm',
    'antecedent_rain_30d_coverage',
    'pre_event_flow_mm_h',
    'peak_flow_mm_h',
    'direct_runoff_lh_mm',
    'direct_runoff_eckhardt_mm',
    'runoff_ratio_lh',
    'runoff_ratio_eckhardt',
    'cn_event_lambda005_lh',
    'cn_event_lambda005_eckhardt',
    'cn_event_lambda020_lh',
    'cn_event_lambda020_eckhardt',
    'primary_rain_threshold',
    'rain_missing',
    'streamflow_missing',
    'snow_or_freeze',
    'strong_regulation',
    'overlapping_response',
    'response_ended_early',
    'mass_balance_qc_lh',
    'mass_balance_qc_eckhardt',
    'baseflow_method_conflict',
    'analysis_eligible_lh',
    'analysis_eligible_eckhardt',
    'analysis_eligible_both_pre_dwd',
    'split',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeseries-root', required=True, type=Path)
    parser.add_argument('--attributes-root', required=True, type=Path)
    parser.add_argument('--basins-root', required=True, type=Path)
    parser.add_argument('--dwd-features', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--basin-limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wet-threshold', type=float, default=0.1)
    parser.add_argument('--inter-event-dry-hours', type=int, default=12)
    parser.add_argument('--minimum-event-rain', type=float, default=10.0)
    parser.add_argument('--primary-event-rain', type=float, default=20.0)
    parser.add_argument('--maximum-response-hours', type=int, default=96)
    parser.add_argument('--minimum-coverage', type=float, default=0.95)
    parser.add_argument('--lh-alpha', type=float, default=0.925)
    parser.add_argument('--lh-passes', type=int, default=3)
    parser.add_argument('--eckhardt-alpha', type=float, default=0.98)
    parser.add_argument('--eckhardt-bfi-max', type=float, default=0.80)
    return parser.parse_args()


def load_basin_sets(basins_root: Path) -> tuple[list[str], set[str]]:
    development: set[str] = set()
    spatial_holdout: set[str] = set()
    for split in ('train', 'val'):
        path = basins_root / f'{split}_basins.txt'
        if not path.is_file():
            raise FileNotFoundError(path)
        development.update(
            line.strip()
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        )
    test_path = basins_root / 'test_basins.txt'
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    spatial_holdout.update(
        line.strip()
        for line in test_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    )
    overlap = development.intersection(spatial_holdout)
    if overlap:
        raise RuntimeError(f'development/test basin overlap: {sorted(overlap)[:5]}')
    return sorted(development | spatial_holdout), spatial_holdout


def split_for_event(
    basin: str, year: int, spatial_holdout: set[str]
) -> str:
    if basin in spatial_holdout:
        return 'double_holdout' if year == 2024 else 'spatial_test'
    if year <= 2021:
        return 'train'
    if year <= 2023:
        return 'validation'
    return 'temporal_test'


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return 'DJF'
    if month in (3, 4, 5):
        return 'MAM'
    if month in (6, 7, 8):
        return 'JJA'
    return 'SON'


@njit(cache=False)
def _lh_segment(flow: np.ndarray, alpha: float, passes: int) -> np.ndarray:
    work = flow.copy()
    size = len(work)
    for pass_index in range(passes):
        if pass_index % 2:
            signal = work[::-1].copy()
        else:
            signal = work.copy()
        quick = np.zeros(size, dtype=np.float64)
        for index in range(1, size):
            value = (
                alpha * quick[index - 1]
                + 0.5
                * (1.0 + alpha)
                * (signal[index] - signal[index - 1])
            )
            if value < 0.0:
                value = 0.0
            if value > signal[index]:
                value = signal[index]
            quick[index] = value
        filtered = signal - quick
        if pass_index % 2:
            work = filtered[::-1].copy()
        else:
            work = filtered
    return work


@njit(cache=False)
def _eckhardt_segment(
    flow: np.ndarray, alpha: float, bfi_max: float
) -> np.ndarray:
    base = np.empty(len(flow), dtype=np.float64)
    base[0] = flow[0]
    denominator = 1.0 - alpha * bfi_max
    for index in range(1, len(flow)):
        value = (
            (1.0 - bfi_max) * alpha * base[index - 1]
            + (1.0 - alpha) * bfi_max * flow[index]
        ) / denominator
        if value < 0.0:
            value = 0.0
        if value > flow[index]:
            value = flow[index]
        base[index] = value
    return base


def _finite_segments(values: np.ndarray) -> list[tuple[int, int]]:
    finite = np.isfinite(values)
    changes = np.diff(np.pad(finite.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def lyne_hollick_baseflow(
    flow: np.ndarray, alpha: float = 0.925, passes: int = 3
) -> np.ndarray:
    """Return Lyne-Hollick baseflow, preserving missing-data gaps."""
    values = np.asarray(flow, dtype=np.float64)
    if not 0.0 < alpha < 1.0 or passes < 1:
        raise ValueError('invalid Lyne-Hollick parameters')
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for start, end in _finite_segments(values):
        segment = np.maximum(values[start:end], 0.0)
        if len(segment) == 1:
            result[start:end] = segment
        else:
            result[start:end] = _lh_segment(segment, alpha, passes)
    return result


def eckhardt_baseflow(
    flow: np.ndarray, alpha: float = 0.98, bfi_max: float = 0.80
) -> np.ndarray:
    """Return Eckhardt baseflow, preserving missing-data gaps."""
    values = np.asarray(flow, dtype=np.float64)
    if not 0.0 < alpha < 1.0 or not 0.0 < bfi_max < 1.0:
        raise ValueError('invalid Eckhardt parameters')
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for start, end in _finite_segments(values):
        segment = np.maximum(values[start:end], 0.0)
        if len(segment) == 1:
            result[start:end] = segment
        else:
            result[start:end] = _eckhardt_segment(
                segment, alpha, bfi_max
            )
    return result


def delineate_rain_events(
    index: pd.DatetimeIndex,
    precipitation: np.ndarray,
    wet_threshold: float = 0.1,
    inter_event_dry_hours: int = 12,
) -> list[tuple[int, int]]:
    """Return inclusive positional rain-event start/end pairs."""
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError('time index must be monotonic and unique')
    rain = np.asarray(precipitation, dtype=np.float64)
    wet = np.flatnonzero(np.isfinite(rain) & (rain >= wet_threshold))
    if not len(wet):
        return []
    breaks = np.flatnonzero(np.diff(wet) > inter_event_dry_hours) + 1
    groups = np.split(wet, breaks)
    return [(int(group[0]), int(group[-1])) for group in groups]


def scs_runoff_mm(
    precipitation_mm: float, curve_number: float, abstraction_ratio: float
) -> float:
    if not 0.0 < curve_number < 100.0:
        raise ValueError('curve_number must be between zero and 100')
    if not 0.0 <= abstraction_ratio < 1.0:
        raise ValueError('abstraction_ratio must be in [0, 1)')
    storage = 25400.0 / curve_number - 254.0
    if precipitation_mm <= abstraction_ratio * storage:
        return 0.0
    numerator = (precipitation_mm - abstraction_ratio * storage) ** 2
    denominator = precipitation_mm + (1.0 - abstraction_ratio) * storage
    return numerator / denominator


def invert_event_cn(
    precipitation_mm: float,
    runoff_mm: float,
    abstraction_ratio: float,
) -> float:
    """Invert diagnostic event CN by bisection without clipping to model bounds."""
    if (
        not math.isfinite(precipitation_mm)
        or not math.isfinite(runoff_mm)
        or precipitation_mm <= 0.0
        or runoff_mm <= 0.0
        or runoff_mm > precipitation_mm
    ):
        return math.nan
    low = 0.01
    high = 99.999
    if scs_runoff_mm(precipitation_mm, high, abstraction_ratio) < runoff_mm:
        return math.nan
    for _ in range(80):
        middle = 0.5 * (low + high)
        simulated = scs_runoff_mm(
            precipitation_mm, middle, abstraction_ratio
        )
        if simulated < runoff_mm:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _window_sum_and_coverage(
    values: np.ndarray, start: int, end: int
) -> tuple[float, float]:
    expected = max(0, end - start)
    if expected == 0:
        return math.nan, 0.0
    window = values[max(start, 0) : max(end, 0)]
    finite = np.isfinite(window)
    coverage = float(finite.sum() / expected)
    return float(np.nansum(window)), coverage


def _early_response_end(
    flow: np.ndarray,
    rain_end: int,
    search_end: int,
    pre_event_flow: float,
    consecutive_hours: int = 6,
) -> int | None:
    if not math.isfinite(pre_event_flow) or search_end <= rain_end:
        return None
    near_base = max(pre_event_flow * 1.10, pre_event_flow + 0.001)
    run = 0
    for index in range(rain_end + 1, search_end + 1):
        if math.isfinite(flow[index]) and flow[index] <= near_base:
            run += 1
            if run == consecutive_hours:
                return index
        else:
            run = 0
    return None


def _method_conflict(runoff_lh: float, runoff_eckhardt: float) -> bool:
    if not math.isfinite(runoff_lh) or not math.isfinite(runoff_eckhardt):
        return True
    larger = max(runoff_lh, runoff_eckhardt)
    smaller = min(runoff_lh, runoff_eckhardt)
    if larger <= 0.1:
        return False
    return smaller <= 0.01 or larger / smaller >= 10.0


def extract_events_from_frame(
    frame: pd.DataFrame,
    basin: str,
    spatial_holdout: set[str],
    strong_regulation: bool,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract event rows for one basin from an hourly CAMELS frame."""
    required = {TIME_COLUMN, RAIN_COLUMN, FLOW_COLUMN, TEMPERATURE_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f'{basin} missing columns: {sorted(missing)}')
    local = frame[list(required)].copy()
    local[TIME_COLUMN] = pd.to_datetime(local[TIME_COLUMN], errors='coerce')
    invalid_time_rows = int(local[TIME_COLUMN].isna().sum())
    local = local.dropna(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN)
    duplicate_time_rows = int(local.duplicated(TIME_COLUMN).sum())
    local = local.drop_duplicates(TIME_COLUMN, keep='first').set_index(TIME_COLUMN)
    if local.empty:
        raise RuntimeError(f'{basin} has no valid timestamps')
    full_index = pd.date_range(local.index.min(), local.index.max(), freq='h')
    missing_time_rows = len(full_index) - len(local)
    local = local.reindex(full_index)
    for column in (RAIN_COLUMN, FLOW_COLUMN, TEMPERATURE_COLUMN):
        local[column] = pd.to_numeric(local[column], errors='coerce')
    rain = local[RAIN_COLUMN].to_numpy(dtype=np.float64)
    flow_observed = local[FLOW_COLUMN].to_numpy(dtype=np.float64)
    temperature = local[TEMPERATURE_COLUMN].to_numpy(dtype=np.float64)
    negative_rain = np.isfinite(rain) & (rain < 0.0)
    negative_flow = np.isfinite(flow_observed) & (flow_observed < 0.0)
    rain[negative_rain] = np.nan
    flow_observed[negative_flow] = np.nan
    flow_for_filter = (
        pd.Series(flow_observed)
        .interpolate(limit=3, limit_area='inside')
        .to_numpy(dtype=np.float64)
    )
    base_lh = lyne_hollick_baseflow(
        flow_for_filter,
        alpha=float(config['lh_alpha']),
        passes=int(config['lh_passes']),
    )
    base_eckhardt = eckhardt_baseflow(
        flow_for_filter,
        alpha=float(config['eckhardt_alpha']),
        bfi_max=float(config['eckhardt_bfi_max']),
    )
    quick_lh = np.maximum(flow_for_filter - base_lh, 0.0)
    quick_eckhardt = np.maximum(flow_for_filter - base_eckhardt, 0.0)
    candidates = delineate_rain_events(
        full_index,
        rain,
        wet_threshold=float(config['wet_threshold']),
        inter_event_dry_hours=int(config['inter_event_dry_hours']),
    )
    rows: list[dict[str, Any]] = []
    below_sensitivity = 0
    for event_index, (start, rain_end) in enumerate(candidates):
        rain_total, rain_coverage = _window_sum_and_coverage(
            rain, start, rain_end + 1
        )
        if rain_total < float(config['minimum_event_rain']):
            below_sensitivity += 1
            continue
        hard_end = min(
            len(full_index) - 1,
            rain_end + int(config['maximum_response_hours']),
        )
        next_start = (
            candidates[event_index + 1][0]
            if event_index + 1 < len(candidates)
            else None
        )
        search_end = hard_end
        if next_start is not None and next_start <= hard_end:
            search_end = max(rain_end, next_start - 1)
        pre_start = max(0, start - 24)
        pre_window = flow_observed[pre_start:start]
        pre_event_flow = (
            float(np.nanmedian(pre_window))
            if np.isfinite(pre_window).any()
            else math.nan
        )
        early_end = _early_response_end(
            flow_observed, rain_end, search_end, pre_event_flow
        )
        response_ended_early = early_end is not None
        response_end = early_end if early_end is not None else search_end
        overlapping = bool(
            next_start is not None
            and next_start <= hard_end
            and early_end is None
        )
        response_slice = slice(start, response_end + 1)
        response_flow = flow_observed[response_slice]
        streamflow_coverage = float(
            np.isfinite(response_flow).sum() / len(response_flow)
        )
        runoff_lh = (
            float(np.nansum(quick_lh[response_slice]))
            if np.isfinite(quick_lh[response_slice]).any()
            else math.nan
        )
        runoff_eckhardt = (
            float(np.nansum(quick_eckhardt[response_slice]))
            if np.isfinite(quick_eckhardt[response_slice]).any()
            else math.nan
        )
        ratio_lh = runoff_lh / rain_total if rain_total > 0.0 else math.nan
        ratio_eckhardt = (
            runoff_eckhardt / rain_total if rain_total > 0.0 else math.nan
        )
        mass_balance_lh = math.isfinite(ratio_lh) and 0.0 <= ratio_lh <= 1.2
        mass_balance_eckhardt = (
            math.isfinite(ratio_eckhardt)
            and 0.0 <= ratio_eckhardt <= 1.2
        )
        rain_missing = rain_coverage < float(config['minimum_coverage'])
        streamflow_missing = (
            streamflow_coverage < float(config['minimum_coverage'])
        )
        rain_temperature = temperature[start : rain_end + 1]
        rain_intensity = rain[start : rain_end + 1]
        precip_temp = rain_temperature[
            np.isfinite(rain_intensity) & (rain_intensity >= 0.1)
        ]
        antecedent_temp = temperature[max(0, start - 72) : start]
        cold_rain_fraction = (
            float(np.mean(precip_temp <= 1.0))
            if np.isfinite(precip_temp).any()
            else 0.0
        )
        antecedent_freeze = bool(
            np.isfinite(antecedent_temp).any()
            and np.nanmean(antecedent_temp) <= 0.0
        )
        snow_or_freeze = cold_rain_fraction >= 0.25 or antecedent_freeze
        conflict = _method_conflict(runoff_lh, runoff_eckhardt)
        common_eligible = not any(
            (
                rain_missing,
                streamflow_missing,
                snow_or_freeze,
                strong_regulation,
                overlapping,
            )
        )
        eligible_lh = common_eligible and mass_balance_lh
        eligible_eckhardt = common_eligible and mass_balance_eckhardt
        timestamp = full_index[start]
        antecedent_5d, antecedent_5d_coverage = _window_sum_and_coverage(
            rain, start - 120, start
        )
        antecedent_30d, antecedent_30d_coverage = _window_sum_and_coverage(
            rain, start - 720, start
        )
        peak_flow = (
            float(np.nanmax(response_flow))
            if np.isfinite(response_flow).any()
            else math.nan
        )
        rows.append(
            {
                'event_id': f'{basin}_{timestamp:%Y%m%dT%H}',
                'basin': basin,
                'gauge_id': basin.removeprefix(MODEL_PREFIX),
                'event_start': timestamp,
                'rain_end': full_index[rain_end],
                'response_end': full_index[response_end],
                'season': season_for_month(timestamp.month),
                'rain_hours': rain_end - start + 1,
                'response_hours': response_end - start + 1,
                'event_rain_mm': rain_total,
                'rain_coverage': rain_coverage,
                'streamflow_coverage': streamflow_coverage,
                'antecedent_rain_5d_mm': antecedent_5d,
                'antecedent_rain_5d_coverage': antecedent_5d_coverage,
                'antecedent_rain_30d_mm': antecedent_30d,
                'antecedent_rain_30d_coverage': antecedent_30d_coverage,
                'pre_event_flow_mm_h': pre_event_flow,
                'peak_flow_mm_h': peak_flow,
                'direct_runoff_lh_mm': runoff_lh,
                'direct_runoff_eckhardt_mm': runoff_eckhardt,
                'runoff_ratio_lh': ratio_lh,
                'runoff_ratio_eckhardt': ratio_eckhardt,
                'cn_event_lambda005_lh': invert_event_cn(
                    rain_total, runoff_lh, 0.05
                ),
                'cn_event_lambda005_eckhardt': invert_event_cn(
                    rain_total, runoff_eckhardt, 0.05
                ),
                'cn_event_lambda020_lh': invert_event_cn(
                    rain_total, runoff_lh, 0.20
                ),
                'cn_event_lambda020_eckhardt': invert_event_cn(
                    rain_total, runoff_eckhardt, 0.20
                ),
                'primary_rain_threshold': rain_total
                >= float(config['primary_event_rain']),
                'rain_missing': rain_missing,
                'streamflow_missing': streamflow_missing,
                'snow_or_freeze': snow_or_freeze,
                'strong_regulation': strong_regulation,
                'overlapping_response': overlapping,
                'response_ended_early': response_ended_early,
                'mass_balance_qc_lh': not mass_balance_lh,
                'mass_balance_qc_eckhardt': not mass_balance_eckhardt,
                'baseflow_method_conflict': conflict,
                'analysis_eligible_lh': eligible_lh,
                'analysis_eligible_eckhardt': eligible_eckhardt,
                'analysis_eligible_both_pre_dwd': eligible_lh
                and eligible_eckhardt
                and not conflict,
                'split': split_for_event(
                    basin, timestamp.year, spatial_holdout
                ),
            }
        )
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    audit = {
        'basin': basin,
        'source_start': str(full_index.min()),
        'source_end': str(full_index.max()),
        'source_hour_count': len(full_index),
        'invalid_time_rows': invalid_time_rows,
        'duplicate_time_rows': duplicate_time_rows,
        'missing_time_rows': missing_time_rows,
        'negative_rain_rows': int(negative_rain.sum()),
        'negative_flow_rows': int(negative_flow.sum()),
        'candidate_storm_count': len(candidates),
        'below_sensitivity_rain_count': below_sensitivity,
        'retained_event_count': len(events),
        'primary_event_count': int(events['primary_rain_threshold'].sum())
        if len(events)
        else 0,
        'eligible_both_pre_dwd_count': int(
            events['analysis_eligible_both_pre_dwd'].sum()
        )
        if len(events)
        else 0,
    }
    return events, audit


def _timeseries_path(root: Path, basin: str) -> Path:
    gauge_id = basin.removeprefix(MODEL_PREFIX)
    return root / f'CAMELS_DE_1h_hydromet_timeseries_{gauge_id}.csv'


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _process_basin(task: dict[str, Any]) -> dict[str, Any]:
    basin = str(task['basin'])
    source = _timeseries_path(Path(task['timeseries_root']), basin)
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(
        source,
        usecols=[TIME_COLUMN, RAIN_COLUMN, FLOW_COLUMN, TEMPERATURE_COLUMN],
    )
    events, audit = extract_events_from_frame(
        frame,
        basin=basin,
        spatial_holdout=set(task['spatial_holdout']),
        strong_regulation=bool(task['strong_regulation']),
        config=dict(task['config']),
    )
    shard = Path(task['shard'])
    audit_path = Path(task['audit_path'])
    events.to_parquet(shard, index=False, compression='zstd')
    audit['source_file'] = str(source.resolve())
    audit['source_bytes'] = source.stat().st_size
    _write_json(audit_path, audit)
    return audit


def load_attributes(
    attributes_root: Path, basins: list[str]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    groups = ('soil', 'landcover', 'topographic', 'humaninfluence')
    merged: pd.DataFrame | None = None
    for group in groups:
        path = attributes_root / f'CAMELS_DE_1h_{group}_attributes.csv'
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if 'gauge_id' not in frame:
            raise RuntimeError(f'{path} lacks gauge_id')
        merged = frame if merged is None else merged.merge(
            frame, on='gauge_id', how='outer', validate='one_to_one'
        )
    assert merged is not None
    merged.insert(0, 'basin', MODEL_PREFIX + merged['gauge_id'].astype(str))
    merged = merged[merged['basin'].isin(basins)].copy()
    if len(merged) != len(basins):
        missing = sorted(set(basins).difference(merged['basin']))
        raise RuntimeError(f'missing attributes for {missing[:10]}')
    area = pd.to_numeric(merged.get('area'), errors='coerce')
    lake_area = pd.to_numeric(
        merged.get('dams_total_lake_area'), errors='coerce'
    ).fillna(0.0)
    lake_volume = pd.to_numeric(
        merged.get('dams_total_lake_volume'), errors='coerce'
    ).fillna(0.0)
    storage_mm = lake_volume / area
    lake_fraction = lake_area / area
    merged['strong_regulation'] = (storage_mm >= 10.0) | (
        lake_fraction >= 0.01
    )
    regulation = dict(
        zip(
            merged['basin'],
            merged['strong_regulation'].astype(bool),
            strict=True,
        )
    )
    return merged.sort_values('basin').reset_index(drop=True), regulation


def join_dwd_features(events: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join only DWD state that was valid at least 48 h before event start."""
    with np.load(path, allow_pickle=False) as payload:
        basin = payload['basin'].astype(str)
        dates = payload['date'].astype('datetime64[D]')
        feature_keys = sorted(
            key
            for key in payload.files
            if key not in {'basin', 'date'}
        )
        features = {key: payload[key] for key in feature_keys}
    if len(np.unique(basin)) != len(basin):
        raise RuntimeError('duplicate basins in DWD feature file')
    if not np.array_equal(dates, np.unique(dates)):
        raise RuntimeError('DWD dates are not unique/sorted')
    basin_lookup = {value: index for index, value in enumerate(basin)}
    event_basin_index = np.array(
        [basin_lookup.get(value, -1) for value in events['basin']],
        dtype=np.int64,
    )
    state_dates = (
        events['event_start'].to_numpy(dtype='datetime64[h]')
        - np.timedelta64(48, 'h')
    ).astype('datetime64[D]')
    date_index = np.searchsorted(dates, state_dates)
    in_bounds = (date_index >= 0) & (date_index < len(dates))
    exact = np.zeros(len(events), dtype=bool)
    exact[in_bounds] = dates[date_index[in_bounds]] == state_dates[in_bounds]
    valid = exact & (event_basin_index >= 0)
    joined = events.copy()
    joined['dwd_state_valid_date'] = state_dates
    for key, values in features.items():
        output = np.full(len(events), np.nan, dtype=np.float32)
        output[valid] = values[
            event_basin_index[valid], date_index[valid]
        ].astype(np.float32)
        joined[key] = output
    history_days = np.zeros(len(events), dtype=np.int32)
    expanding_p20 = np.full(len(events), np.nan, dtype=np.float32)
    primary_key = 'dwd_soil_0_30_p50'
    for basin_name, row_index in joined.groupby('basin', sort=False).groups.items():
        basin_position = basin_lookup.get(str(basin_name))
        if basin_position is None:
            continue
        row_positions = np.asarray(list(row_index), dtype=np.int64)
        series = features[primary_key][basin_position].astype(np.float64)
        for row_position in row_positions:
            position = date_index[row_position]
            if not valid[row_position]:
                continue
            history = series[: position + 1]
            finite = history[np.isfinite(history)]
            history_days[row_position] = len(finite)
            if len(finite) >= 730:
                expanding_p20[row_position] = np.float32(
                    np.quantile(finite, 0.20)
                )
    joined['dwd_history_days'] = history_days
    joined['dwd_soil_0_30_expanding_p20'] = expanding_p20
    joined['dwd_extreme_dry_expanding'] = (
        joined[primary_key] < joined['dwd_soil_0_30_expanding_p20']
    ) & joined[primary_key].notna()
    joined['dwd_extreme_dry_raw_lt20'] = joined[primary_key] < 20.0
    joined['dwd_available'] = joined[primary_key].notna()
    joined['analysis_eligible_both'] = (
        joined['analysis_eligible_both_pre_dwd']
        & joined['dwd_available']
        & joined['primary_rain_threshold']
    )
    return joined


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dry_sample_audit(open_events: pd.DataFrame) -> dict[str, Any]:
    eligible = open_events[open_events['analysis_eligible_both']].copy()
    if eligible.empty:
        return {
            'eligible_event_count': 0,
            'eligible_basin_count': 0,
            'expanding_dry_event_count': 0,
            'expanding_dry_basin_count': 0,
            'expanding_dry_top_decile_event_count': 0,
        }
    threshold = eligible.groupby('basin')['direct_runoff_lh_mm'].transform(
        lambda values: values.quantile(0.90)
    )
    eligible['basin_top_decile_runoff'] = (
        eligible['direct_runoff_lh_mm'] >= threshold
    )
    dry = eligible[eligible['dwd_extreme_dry_expanding']]
    return {
        'eligible_event_count': len(eligible),
        'eligible_basin_count': int(eligible['basin'].nunique()),
        'expanding_dry_event_count': len(dry),
        'expanding_dry_basin_count': int(dry['basin'].nunique()),
        'expanding_dry_top_decile_event_count': int(
            dry['basin_top_decile_runoff'].sum()
        ),
        'm2_minimum_gate': {
            'events': 500,
            'basins': 100,
            'top_decile_events': 100,
        },
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError('workers must be in [1, 8]')
    if args.minimum_event_rain > args.primary_event_rain:
        raise ValueError('minimum-event-rain cannot exceed primary-event-rain')
    output_root = args.output_root.resolve()
    completed_manifest = output_root / 'artifact_manifest.json'
    if completed_manifest.is_file():
        raise RuntimeError(
            'output-root is already complete; refuse to overwrite or resume it'
        )
    if output_root.exists() and not args.resume:
        raise RuntimeError('output-root exists; pass --resume to resume it')
    output_root.mkdir(parents=True, exist_ok=True)
    shards = output_root / 'basin_events'
    audits_root = output_root / 'basin_audits'
    double_root = output_root / 'double_holdout'
    for directory in (shards, audits_root, double_root):
        directory.mkdir(exist_ok=True)
    basins, spatial_holdout = load_basin_sets(args.basins_root)
    if args.basin_limit is not None:
        if args.basin_limit < 1:
            raise ValueError('basin-limit must be positive')
        basins = basins[: args.basin_limit]
        spatial_holdout = spatial_holdout.intersection(basins)
    attributes, regulation = load_attributes(args.attributes_root, basins)
    config = {
        'wet_threshold': args.wet_threshold,
        'inter_event_dry_hours': args.inter_event_dry_hours,
        'minimum_event_rain': args.minimum_event_rain,
        'primary_event_rain': args.primary_event_rain,
        'maximum_response_hours': args.maximum_response_hours,
        'minimum_coverage': args.minimum_coverage,
        'lh_alpha': args.lh_alpha,
        'lh_passes': args.lh_passes,
        'eckhardt_alpha': args.eckhardt_alpha,
        'eckhardt_bfi_max': args.eckhardt_bfi_max,
    }
    # Compile the numerical kernels before forking workers.
    _lh_segment(np.ones(4, dtype=np.float64), args.lh_alpha, args.lh_passes)
    _eckhardt_segment(
        np.ones(4, dtype=np.float64),
        args.eckhardt_alpha,
        args.eckhardt_bfi_max,
    )
    tasks: list[dict[str, Any]] = []
    basin_audits: list[dict[str, Any]] = []
    for basin in basins:
        shard = shards / f'{basin}.parquet'
        audit_path = audits_root / f'{basin}.json'
        if args.resume and shard.is_file() and audit_path.is_file():
            basin_audits.append(
                json.loads(audit_path.read_text(encoding='utf-8'))
            )
            continue
        tasks.append(
            {
                'basin': basin,
                'timeseries_root': str(args.timeseries_root.resolve()),
                'spatial_holdout': sorted(spatial_holdout),
                'strong_regulation': regulation[basin],
                'config': config,
                'shard': str(shard),
                'audit_path': str(audit_path),
            }
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_basin, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            audit = future.result()
            basin_audits.append(audit)
            completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            'completed_new_basins': completed,
                            'total_new_basins': len(tasks),
                            'basin': audit['basin'],
                            'events': audit['retained_event_count'],
                        }
                    ),
                    flush=True,
                )
    shard_paths = [shards / f'{basin}.parquet' for basin in basins]
    missing_shards = [str(path) for path in shard_paths if not path.is_file()]
    if missing_shards:
        raise RuntimeError(f'missing event shards: {missing_shards[:5]}')
    events = pd.concat(
        (pd.read_parquet(path) for path in shard_paths), ignore_index=True
    )
    events = events.sort_values(['basin', 'event_start']).reset_index(drop=True)
    events = join_dwd_features(events, args.dwd_features)
    open_events = events[events['split'] != 'double_holdout'].copy()
    double_holdout = events[events['split'] == 'double_holdout'].copy()
    open_path = output_root / 'events_open.parquet'
    double_path = double_root / 'events_sealed.parquet'
    attributes_path = output_root / 'basin_attributes.parquet'
    audit_path = output_root / 'basin_build_audit.parquet'
    open_events.to_parquet(open_path, index=False, compression='zstd')
    double_holdout.to_parquet(double_path, index=False, compression='zstd')
    attributes.to_parquet(attributes_path, index=False, compression='zstd')
    pd.DataFrame(basin_audits).sort_values('basin').to_parquet(
        audit_path, index=False, compression='zstd'
    )
    sealed_resume_shards = double_root / 'resume_shards_mixed'
    if sealed_resume_shards.exists():
        raise RuntimeError(
            f'sealed resume-shard destination exists: {sealed_resume_shards}'
        )
    shards.replace(sealed_resume_shards)
    sealed_shard_paths = sorted(sealed_resume_shards.glob('*.parquet'))
    summary = {
        'schema': 'germany-dynamic-cn-event-catalog-v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'complete_national_build': args.basin_limit is None,
        'basin_count': len(basins),
        'development_basin_count': len(set(basins).difference(spatial_holdout)),
        'spatial_holdout_basin_count': len(spatial_holdout),
        'source_period': [
            min(audit['source_start'] for audit in basin_audits),
            max(audit['source_end'] for audit in basin_audits),
        ],
        'event_contract': config,
        'open_event_count': len(open_events),
        'double_holdout_event_count_sealed': len(double_holdout),
        'open_split_counts': {
            str(key): int(value)
            for key, value in open_events['split'].value_counts().items()
        },
        'open_primary_event_count': int(
            open_events['primary_rain_threshold'].sum()
        ),
        'open_analysis_eligible_both_count': int(
            open_events['analysis_eligible_both'].sum()
        ),
        'dry_sample_audit_open_only': _dry_sample_audit(open_events),
        'double_holdout_policy': (
            'Targets and mixed recovery shards are physically confined to the '
            'double_holdout directory. Do not read target columns until the '
            'preregistered final evaluation.'
        ),
        'sealed_resume_shards': {
            'count': len(sealed_shard_paths),
            'bytes': sum(path.stat().st_size for path in sealed_shard_paths),
            'path': 'double_holdout/resume_shards_mixed',
        },
        'static_regulation_definition': (
            'dams_total_lake_volume / basin_area >= 10 mm or '
            'dams_total_lake_area / basin_area >= 1%'
        ),
        'snow_freeze_definition': (
            'at least 25% of wet hours have T<=1C, or mean prior-72h T<=0C'
        ),
        'dwd_availability_lag_hours': 48,
        'diagnostic_cn_note': (
            'Derived from observed P and baseflow-separated Q only for QA; '
            'never an independent supervised ground truth.'
        ),
    }
    summary_path = output_root / 'build_summary.json'
    _write_json(summary_path, summary)
    contract = {
        'source_timeseries_root': str(args.timeseries_root.resolve()),
        'source_attributes_root': str(args.attributes_root.resolve()),
        'source_basins_root': str(args.basins_root.resolve()),
        'source_dwd_features': str(args.dwd_features.resolve()),
        'builder': str(Path(__file__).resolve()),
        'config': config,
    }
    contract_path = output_root / 'build_contract.json'
    _write_json(contract_path, contract)
    manifest_files = [
        open_path,
        double_path,
        attributes_path,
        audit_path,
        summary_path,
        contract_path,
    ]
    manifest = {
        'schema': 'germany-dynamic-cn-event-catalog-manifest-v1',
        'files': [
            {
                'path': str(path.relative_to(output_root)),
                'bytes': path.stat().st_size,
                'sha256': sha256(path),
            }
            for path in manifest_files
        ],
    }
    _write_json(output_root / 'artifact_manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
