#!/usr/bin/env python3
"""Retrospective open-split event confirmation for the dry-soil CN hypothesis.

This analysis does not open or inspect the sealed spatial-2024 double holdout.
It enriches the frozen severe-dry candidates with hourly rainfall/flow shape,
causal DWD drought-memory diagnostics, and same-basin/same-season wet controls.
The result is evidence about event consistency, not observed CN ground truth or
an unbiased estimate of the causal effect of dry soil.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from train_dynamic_cn_models import prepare_model_frame, verify_open_input

RAIN_COLUMN = 'precipitation_mean_gapfilled'
FLOW_COLUMN = 'discharge_spec_obs'
TIME_COLUMN = 'date'
HOURLY_MATCH_FEATURES = (
    'event_rain_mm',
    'rain_hours',
    'rain_max_1h_mm',
    'rain_max_3h_mm',
    'rain_max_6h_mm',
)
MATCH_RATIO_BOUNDS = {
    'event_rain_mm': (0.75, 4.0 / 3.0),
    'rain_hours': (0.50, 2.00),
    'rain_max_1h_mm': (0.50, 2.00),
    'rain_max_3h_mm': (0.50, 2.00),
    'rain_max_6h_mm': (0.50, 2.00),
}
MINIMUM_CONTROL_WETNESS_GAP_NFK = 10.0
NAMED_AACHEN_EVENT = 'camelsde1h_DEA11390_20140609T19'
NAMED_ALLGAEU_EVENTS = {
    'camelsde1h_DE210300_20220819T10',
    'camelsde1h_DE210320_20220819T13',
    'camelsde1h_DE210360_20220819T10',
    'camelsde1h_DE210760_20220819T10',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--timeseries-root', required=True, type=Path)
    parser.add_argument('--dwd-features', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_compatible(value), indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def reject_sealed_path(path: Path) -> None:
    if any(part.lower() == 'double_holdout' for part in path.parts):
        raise RuntimeError(f'sealed double-holdout path rejected: {path}')


def source_record(path: Path, include_hash: bool = True) -> dict[str, Any]:
    reject_sealed_path(path)
    record: dict[str, Any] = {
        'path': str(path.resolve()),
        'bytes': path.stat().st_size,
    }
    if include_hash:
        record['sha256'] = sha256(path)
    return record


def effective_dry_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Select frozen M2 dry events without conditioning on runoff outcome."""
    basin_q90 = frame.groupby('basin')['direct_runoff_lh_mm'].transform(
        lambda values: values.quantile(0.90)
    )
    selected = frame[frame['m2_effective_dry']].copy()
    selected['basin_q90_direct_runoff_lh_mm'] = basin_q90[selected.index]
    selected['severe_top_decile_lh'] = (
        selected['direct_runoff_lh_mm']
        >= selected['basin_q90_direct_runoff_lh_mm']
    )
    selected['strict_open_holdout'] = selected['split'].isin(
        ['spatial_test', 'temporal_test']
    )
    selected['named_aachen_case'] = selected['event_id'].eq(NAMED_AACHEN_EVENT)
    selected['named_allgaeu_case'] = selected['event_id'].isin(
        NAMED_ALLGAEU_EVENTS
    )
    return selected.sort_values(['basin', 'event_start']).reset_index(drop=True)


def add_control_state_classes(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify controls from outcome-blind train state-season quantiles."""
    output = frame.copy()
    train = output[output['split'] == 'train']
    grouped = train.groupby(['federal_state', 'season'], observed=True)[
        'dwd_soil_0_30_p50'
    ]
    quantiles = grouped.quantile([0.35, 0.65, 0.80]).unstack()
    quantiles.columns = [
        'control_state_p35',
        'control_state_p65',
        'control_state_p80',
    ]
    output = output.merge(
        quantiles,
        left_on=['federal_state', 'season'],
        right_index=True,
        how='left',
        validate='many_to_one',
    )
    output['control_intermediate'] = (
        output['dwd_soil_0_30_p50'].between(
            output['control_state_p35'], output['control_state_p65']
        )
        & ~output['dry_model']
    )
    output['control_high_wetness'] = (
        output['dwd_soil_0_30_p50'] >= output['control_state_p80']
    ) & ~output['dry_model']
    return output


def preliminary_control_pool(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    control_class: str,
) -> pd.DataFrame:
    """Find outcome-blind wet controls before reading hourly hydrographs."""
    control_column = f'control_{control_class}'
    if control_column not in frame:
        raise ValueError(f'unknown control class: {control_class}')
    pools: list[pd.DataFrame] = []
    by_basin_season_split = {
        key: group
        for key, group in frame[frame[control_column]].groupby(
            ['basin', 'season', 'split'], observed=True, sort=False
        )
    }
    for candidate in candidates.itertuples(index=False):
        controls = by_basin_season_split.get(
            (candidate.basin, candidate.season, candidate.split)
        )
        if controls is None:
            continue
        rain_ratio = controls['event_rain_mm'] / candidate.event_rain_mm
        duration_ratio = controls['rain_hours'] / candidate.rain_hours
        wetness_gap = (
            controls['dwd_soil_0_30_p50'] - candidate.dwd_soil_0_30_p50
        )
        eligible = controls[
            rain_ratio.between(*MATCH_RATIO_BOUNDS['event_rain_mm'])
            & duration_ratio.between(*MATCH_RATIO_BOUNDS['rain_hours'])
            & (wetness_gap >= MINIMUM_CONTROL_WETNESS_GAP_NFK)
        ].copy()
        if eligible.empty:
            continue
        eligible.insert(0, 'candidate_event_id', candidate.event_id)
        pools.append(eligible)
    if not pools:
        return pd.DataFrame(columns=['candidate_event_id', *frame.columns])
    return pd.concat(pools, ignore_index=True)


def rolling_max_sum(values: np.ndarray, hours: int) -> float:
    if not len(values):
        return math.nan
    required = min(hours, len(values))
    rolling = pd.Series(values).rolling(hours, min_periods=required).sum()
    return float(rolling.max()) if rolling.notna().any() else math.nan


def extract_hourly_features(
    events: pd.DataFrame, timeseries_root: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Extract rainfall morphology and raw-flow timing for selected events."""
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for basin, basin_events in events.groupby('basin', sort=True):
        gauge_id = str(basin).removeprefix('camelsde1h_')
        source = timeseries_root / (
            f'CAMELS_DE_1h_hydromet_timeseries_{gauge_id}.csv'
        )
        reject_sealed_path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        raw = pd.read_csv(
            source, usecols=[TIME_COLUMN, RAIN_COLUMN, FLOW_COLUMN]
        )
        raw[TIME_COLUMN] = pd.to_datetime(raw[TIME_COLUMN], errors='coerce')
        raw = (
            raw.dropna(subset=[TIME_COLUMN])
            .drop_duplicates(TIME_COLUMN, keep='first')
            .sort_values(TIME_COLUMN)
            .set_index(TIME_COLUMN)
        )
        for column in (RAIN_COLUMN, FLOW_COLUMN):
            raw[column] = pd.to_numeric(raw[column], errors='coerce')
            raw.loc[raw[column] < 0.0, column] = np.nan
        sources.append(
            {
                'path': str(source.resolve()),
                'bytes': source.stat().st_size,
                'mtime_ns': source.stat().st_mtime_ns,
            }
        )
        for event in basin_events.itertuples(index=False):
            rain_window = raw.loc[event.event_start : event.rain_end]
            response_window = raw.loc[event.event_start : event.response_end]
            rain = rain_window[RAIN_COLUMN].to_numpy(dtype=float)
            flow = response_window[FLOW_COLUMN].to_numpy(dtype=float)
            if len(rain_window) != int(event.rain_hours):
                raise RuntimeError(
                    f'{event.event_id} hourly rain-window length mismatch'
                )
            rain_total = float(np.nansum(rain))
            if not math.isclose(
                rain_total,
                float(event.event_rain_mm),
                rel_tol=0.0,
                abs_tol=1e-3,
            ):
                raise RuntimeError(
                    f'{event.event_id} rain total changed: '
                    f'{rain_total} != {event.event_rain_mm}'
                )
            finite_flow = np.isfinite(flow)
            peak_position = (
                int(np.nanargmax(flow)) if finite_flow.any() else None
            )
            peak_flow = (
                float(flow[peak_position])
                if peak_position is not None
                else math.nan
            )
            if math.isfinite(float(event.peak_flow_mm_h)) and not math.isclose(
                peak_flow,
                float(event.peak_flow_mm_h),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise RuntimeError(f'{event.event_id} peak flow changed')
            rain_nonnegative = np.where(np.isfinite(rain), rain, 0.0)
            offsets = np.arange(len(rain_nonnegative), dtype=float)
            centroid = (
                float(np.dot(offsets, rain_nonnegative) / rain_total)
                if rain_total > 0.0
                else math.nan
            )
            baseline = float(event.pre_event_flow_mm_h)
            onset_threshold = max(baseline * 1.10, baseline + 0.001)
            onset_position: int | None = None
            above = np.isfinite(flow) & (flow > onset_threshold)
            if len(above) >= 2:
                starts = np.flatnonzero(above[:-1] & above[1:])
                if len(starts):
                    onset_position = int(starts[0])
            max_1h = (
                float(np.nanmax(rain)) if np.isfinite(rain).any() else math.nan
            )
            max_3h = rolling_max_sum(rain, 3)
            max_6h = rolling_max_sum(rain, 6)
            rows.append(
                {
                    'event_id': event.event_id,
                    'rain_max_1h_mm': max_1h,
                    'rain_max_3h_mm': max_3h,
                    'rain_max_6h_mm': max_6h,
                    'rain_max_3h_fraction': max_3h / rain_total,
                    'rain_centroid_hours_after_start': centroid,
                    'flow_onset_hours_after_start': onset_position,
                    'flow_peak_hours_after_start': peak_position,
                    'rain_centroid_to_flow_peak_hours': (
                        peak_position - centroid
                        if peak_position is not None
                        else math.nan
                    ),
                    'flow_peak_above_baseline_mm_h': peak_flow - baseline,
                    'peak_flow_per_event_rain_h_inv': peak_flow / rain_total,
                }
            )
    features = pd.DataFrame(rows).drop_duplicates('event_id')
    return features, sources


def add_dwd_memory(events: pd.DataFrame, dwd_path: Path) -> pd.DataFrame:
    """Add causal trailing DWD diagnostics ending on dwd_state_valid_date."""
    reject_sealed_path(dwd_path)
    with np.load(dwd_path, allow_pickle=False) as payload:
        basins = payload['basin'].astype(str)
        dates = payload['date'].astype('datetime64[D]')
        soil_010 = payload['dwd_soil_0_10_p50']
        soil_030 = payload['dwd_soil_0_30_p50']
        basin_lookup = {basin: index for index, basin in enumerate(basins)}
        rows: list[dict[str, Any]] = []
        for event in events.itertuples(index=False):
            basin_index = basin_lookup.get(event.basin)
            valid_date = np.datetime64(event.dwd_state_valid_date, 'D')
            date_index = int(np.searchsorted(dates, valid_date))
            if (
                basin_index is None
                or date_index >= len(dates)
                or dates[date_index] != valid_date
            ):
                raise RuntimeError(f'{event.event_id} lacks causal DWD state')
            record: dict[str, Any] = {'event_id': event.event_id}
            for depth, values in (('0_10', soil_010), ('0_30', soil_030)):
                for days in (7, 30, 60):
                    start = max(0, date_index - days + 1)
                    trailing = values[basin_index, start : date_index + 1]
                    finite = np.isfinite(trailing)
                    record[f'dwd_soil_{depth}_trailing_{days}d_coverage'] = (
                        float(finite.mean())
                    )
                    record[f'dwd_soil_{depth}_trailing_{days}d_mean'] = (
                        float(np.nanmean(trailing))
                        if finite.any()
                        else math.nan
                    )
                    record[f'dwd_soil_{depth}_trailing_{days}d_min'] = (
                        float(np.nanmin(trailing)) if finite.any() else math.nan
                    )
            current = float(soil_030[basin_index, date_index])
            record['dwd_soil_0_30_deficit_to_threshold'] = (
                float(event.dry_threshold_model) - current
            )
            consecutive = 0
            for value in soil_030[basin_index, : date_index + 1][::-1]:
                if (
                    not math.isfinite(float(value))
                    or value >= event.dry_threshold_model
                ):
                    break
                consecutive += 1
            record['dwd_soil_0_30_consecutive_days_below_event_threshold'] = (
                consecutive
            )
            rows.append(record)
    return events.merge(
        pd.DataFrame(rows), on='event_id', how='left', validate='one_to_one'
    )


def prediction_panel(path: Path) -> pd.DataFrame:
    reject_sealed_path(path)
    predictions = pd.read_parquet(path)
    if 'double_holdout' in set(predictions['split']):
        raise RuntimeError('prediction table contains double-holdout rows')
    selected = predictions[
        predictions['model'].isin(['B1', 'M1'])
        & predictions['target'].isin(['lh', 'eckhardt'])
        & predictions['initial_abstraction_ratio'].isin([0.05, 0.20])
    ].copy()
    selected['key'] = (
        selected['model'].str.lower()
        + '_'
        + selected['target']
        + '_lambda'
        + selected['initial_abstraction_ratio'].map({0.05: '005', 0.20: '020'})
    )
    predicted = selected.pivot(
        index='event_id', columns='key', values='predicted_direct_runoff_mm'
    ).add_prefix('predicted_runoff_')
    curve_number = selected.pivot(
        index='event_id', columns='key', values='predicted_curve_number'
    ).add_prefix('predicted_cn_')
    return predicted.join(curve_number).reset_index()


def select_matches(
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    enriched_candidates = candidates.merge(
        hourly, on='event_id', how='left', validate='one_to_one'
    )
    enriched_pool = pool.merge(
        hourly, on='event_id', how='left', validate='many_to_one'
    )
    pairs: list[dict[str, Any]] = []
    for candidate in enriched_candidates.itertuples(index=False):
        possible = enriched_pool[
            enriched_pool['candidate_event_id'] == candidate.event_id
        ].copy()
        if possible.empty:
            pairs.append(
                {
                    'candidate_event_id': candidate.event_id,
                    'control_event_id': None,
                    'match_status': (
                        'NO_SAME_BASIN_SEASON_TOTAL_DURATION_CONTROL'
                    ),
                }
            )
            continue
        valid = np.ones(len(possible), dtype=bool)
        log_ratios: list[np.ndarray] = []
        for feature in HOURLY_MATCH_FEATURES:
            candidate_value = float(getattr(candidate, feature))
            ratios = possible[feature].to_numpy(float) / candidate_value
            lower, upper = MATCH_RATIO_BOUNDS[feature]
            valid &= np.isfinite(ratios) & (ratios >= lower) & (ratios <= upper)
            log_ratios.append(np.log(ratios))
        possible = possible.loc[valid].copy()
        if possible.empty:
            pairs.append(
                {
                    'candidate_event_id': candidate.event_id,
                    'control_event_id': None,
                    'match_status': 'NO_HYETOGRAPH_QUALITY_MATCH',
                }
            )
            continue
        distance_matrix = np.column_stack(
            [values[valid] for values in log_ratios]
        )
        possible['match_log_rms_distance'] = np.sqrt(
            np.mean(distance_matrix**2, axis=1)
        )
        chosen = possible.sort_values(
            ['match_log_rms_distance', 'event_id']
        ).iloc[0]
        record: dict[str, Any] = {
            'candidate_event_id': candidate.event_id,
            'control_event_id': chosen['event_id'],
            'match_status': 'MATCHED',
            'match_log_rms_distance': float(chosen['match_log_rms_distance']),
            'available_quality_controls': len(possible),
        }
        for feature in HOURLY_MATCH_FEATURES:
            record[f'{feature}_ratio_control_to_candidate'] = float(
                chosen[feature] / getattr(candidate, feature)
            )
        pairs.append(record)
    return pd.DataFrame(pairs)


def flatten_pairs(
    matches: pd.DataFrame, enriched: pd.DataFrame
) -> pd.DataFrame:
    candidate_columns = {
        column: f'candidate_{column}' for column in enriched.columns
    }
    control_columns = {
        column: f'control_{column}' for column in enriched.columns
    }
    output = matches.merge(
        enriched.rename(columns=candidate_columns),
        left_on='candidate_event_id',
        right_on='candidate_event_id',
        how='left',
        validate='one_to_one',
    )
    output = output.merge(
        enriched.rename(columns=control_columns),
        left_on='control_event_id',
        right_on='control_event_id',
        how='left',
        validate='many_to_one',
    )
    for name in (
        'runoff_ratio_lh',
        'runoff_ratio_eckhardt',
        'direct_runoff_lh_mm',
        'direct_runoff_eckhardt_mm',
        'cn_event_lambda005_lh',
        'cn_event_lambda005_eckhardt',
        'flow_peak_hours_after_start',
        'peak_flow_per_event_rain_h_inv',
    ):
        output[f'delta_{name}_candidate_minus_control'] = (
            output[f'candidate_{name}'] - output[f'control_{name}']
        )
    output['candidate_b1_lh_lambda005_residual_mm'] = (
        output['candidate_direct_runoff_lh_mm']
        - output['candidate_predicted_runoff_b1_lh_lambda005']
    )
    output['control_b1_lh_lambda005_residual_mm'] = (
        output['control_direct_runoff_lh_mm']
        - output['control_predicted_runoff_b1_lh_lambda005']
    )
    output['supports_reversal_lh'] = (
        output['delta_runoff_ratio_lh_candidate_minus_control'] > 0.0
    ).where(output['match_status'].eq('MATCHED'))
    output['supports_reversal_eckhardt'] = (
        output['delta_runoff_ratio_eckhardt_candidate_minus_control'] > 0.0
    ).where(output['match_status'].eq('MATCHED'))
    return output


def bootstrap_basin_median(
    by_basin: pd.Series, seed: int = 20260805, replicates: int = 5000
) -> list[float]:
    values = by_basin.to_numpy(float)
    if not len(values):
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        samples[index] = np.median(
            values[rng.integers(0, len(values), len(values))]
        )
    return np.quantile(samples, [0.025, 0.975]).tolist()


def summarize_group(frame: pd.DataFrame) -> dict[str, Any]:
    matched = frame[frame['match_status'] == 'MATCHED'].copy()
    result: dict[str, Any] = {
        'candidate_events': len(frame),
        'candidate_basins': int(frame['candidate_basin'].nunique()),
        'matched_events': len(matched),
        'matched_basins': int(matched['candidate_basin'].nunique()),
        'unmatched_events': len(frame) - len(matched),
    }
    for method in ('lh', 'eckhardt'):
        column = f'delta_runoff_ratio_{method}_candidate_minus_control'
        values = matched[column].dropna()
        by_basin = matched.groupby('candidate_basin')[column].mean().dropna()
        positive_basins = int((by_basin > 0.0).sum())
        nonzero_basins = int((by_basin != 0.0).sum())
        result[method] = {
            'positive_event_pairs': int((values > 0.0).sum()),
            'negative_event_pairs': int((values < 0.0).sum()),
            'median_event_pair_delta_runoff_ratio': (
                float(values.median()) if len(values) else math.nan
            ),
            'mean_within_basin_delta_runoff_ratio': (
                float(by_basin.mean()) if len(by_basin) else math.nan
            ),
            'median_within_basin_delta_runoff_ratio': (
                float(by_basin.median()) if len(by_basin) else math.nan
            ),
            'basin_bootstrap_95ci_median_delta': bootstrap_basin_median(
                by_basin
            ),
            'positive_basins': positive_basins,
            'negative_basins': int((by_basin < 0.0).sum()),
            'two_sided_basin_sign_test_p': (
                float(binomtest(positive_basins, nonzero_basins, 0.5).pvalue)
                if nonzero_basins
                else math.nan
            ),
        }
    residual = matched['candidate_b1_lh_lambda005_residual_mm'].dropna()
    result['b1_lh_lambda005'] = {
        'candidate_underpredicted_count': int((residual > 0.0).sum()),
        'candidate_severely_underpredicted_count': int(
            (
                matched['candidate_predicted_runoff_b1_lh_lambda005']
                <= 0.5 * matched['candidate_direct_runoff_lh_mm']
            ).sum()
        ),
        'median_candidate_residual_mm': (
            float(residual.median()) if len(residual) else math.nan
        ),
    }
    return result


def main() -> None:
    args = parse_args()
    for path in (
        args.dataset_root,
        args.timeseries_root,
        args.dwd_features,
        args.predictions,
        args.output_root,
    ):
        reject_sealed_path(path)
    verified_open = verify_open_input(args.dataset_root)
    frame, model_audit = prepare_model_frame(args.dataset_root)
    frame = add_control_state_classes(frame)
    candidates = effective_dry_candidates(frame)
    if len(candidates) != 2627:
        raise RuntimeError(
            f'frozen effective-dry count changed: {len(candidates)}'
        )
    if int(candidates['severe_top_decile_lh'].sum()) != 67:
        raise RuntimeError('frozen severe candidate count changed')
    if int(candidates['strict_open_holdout'].sum()) != 438:
        raise RuntimeError('strict open effective-dry count changed')
    if (
        int(
            (
                candidates['strict_open_holdout']
                & candidates['severe_top_decile_lh']
            ).sum()
        )
        != 14
    ):
        raise RuntimeError('strict open severe candidate count changed')
    if int(candidates['named_aachen_case'].sum()) != 1:
        raise RuntimeError('named Aachen event missing or duplicated')
    if int(candidates['named_allgaeu_case'].sum()) != 4:
        raise RuntimeError('named Allgaeu events missing or duplicated')
    pools = {
        control_class: preliminary_control_pool(
            frame, candidates, control_class
        )
        for control_class in ('intermediate', 'high_wetness')
    }
    selected_ids = set(candidates['event_id'])
    for pool in pools.values():
        selected_ids.update(pool['event_id'])
    selected = frame[frame['event_id'].isin(selected_ids)].copy()
    hourly, hourly_sources = extract_hourly_features(
        selected, args.timeseries_root
    )
    matches_by_class = {
        control_class: select_matches(candidates, pool, hourly)
        for control_class, pool in pools.items()
    }
    matched_control_ids: set[str] = set()
    for matches in matches_by_class.values():
        matched_control_ids.update(
            matches.loc[
                matches['match_status'] == 'MATCHED', 'control_event_id'
            ]
        )
    final_ids = set(candidates['event_id']) | matched_control_ids
    enriched = frame[frame['event_id'].isin(final_ids)].copy()
    candidate_labels = candidates[
        [
            'event_id',
            'severe_top_decile_lh',
            'strict_open_holdout',
            'named_aachen_case',
            'named_allgaeu_case',
        ]
    ]
    enriched = enriched.merge(
        candidate_labels,
        on='event_id',
        how='left',
        validate='one_to_one',
    )
    for column in (
        'severe_top_decile_lh',
        'strict_open_holdout',
        'named_aachen_case',
        'named_allgaeu_case',
    ):
        enriched[column] = enriched[column].eq(True)
    enriched = enriched.merge(
        hourly, on='event_id', how='left', validate='one_to_one'
    )
    attributes = pd.read_parquet(args.dataset_root / 'basin_attributes.parquet')
    enriched = enriched.merge(
        attributes[
            [
                'basin',
                'gauge_name',
                'water_body_name',
                'gauge_lat',
                'gauge_lon',
            ]
        ],
        on='basin',
        how='left',
        validate='many_to_one',
    )
    enriched = add_dwd_memory(enriched, args.dwd_features)
    enriched = enriched.merge(
        prediction_panel(args.predictions),
        on='event_id',
        how='left',
        validate='one_to_one',
    )
    pair_frames = []
    for control_class, matches in matches_by_class.items():
        pair_frame = flatten_pairs(matches, enriched)
        pair_frame.insert(0, 'control_state_comparison', control_class)
        pair_frames.append(pair_frame)
    pairs = pd.concat(pair_frames, ignore_index=True)
    intermediate = pairs[pairs['control_state_comparison'] == 'intermediate']
    high_wetness = pairs[pairs['control_state_comparison'] == 'high_wetness']
    groups = {
        'all_effective_dry_vs_intermediate': intermediate,
        'strict_open_effective_dry_vs_intermediate': intermediate[
            intermediate['candidate_strict_open_holdout']
        ],
        'all_effective_dry_vs_high_wetness': high_wetness,
        'strict_open_effective_dry_vs_high_wetness': high_wetness[
            high_wetness['candidate_strict_open_holdout']
        ],
        'severe_top_decile_vs_intermediate_stress_test': intermediate[
            intermediate['candidate_severe_top_decile_lh']
        ],
        'strict_open_severe_vs_intermediate_stress_test': intermediate[
            intermediate['candidate_strict_open_holdout']
            & intermediate['candidate_severe_top_decile_lh']
        ],
        'named_aachen_2014_vs_intermediate': intermediate[
            intermediate['candidate_named_aachen_case']
        ],
        'named_allgaeu_2022_vs_intermediate': intermediate[
            intermediate['candidate_named_allgaeu_case']
        ],
    }
    summary = {
        'schema': 'dynamic-cn-open-event-confirmation-summary-v2',
        'generated_at_utc': datetime.now(UTC).isoformat(),
        'status': 'RETROSPECTIVE_OPEN_SPLIT_DIAGNOSTIC_NOT_CAUSAL_CONFIRMATION',
        'double_holdout_rows_read': 0,
        'effective_dry_candidate_count': len(candidates),
        'severe_top_decile_stress_test_count': int(
            candidates['severe_top_decile_lh'].sum()
        ),
        'preliminary_control_rows': {
            name: len(pool) for name, pool in pools.items()
        },
        'hourly_events_examined': len(hourly),
        'groups': {
            name: summarize_group(group) for name, group in groups.items()
        },
        'interpretation_rule': (
            'A dry-end reversal would be event-consistent only if '
            'effective-dry '
            'events exceed matched intermediate-state controls across both '
            'baseflow methods and the strict open spatial/temporal subset. The '
            'high-wetness arm diagnoses conventional saturation behavior. This '
            'retrospective screen cannot establish a causal soil effect.'
        ),
    }
    contract = {
        'schema': 'dynamic-cn-open-event-confirmation-contract-v2',
        'created_at_utc': datetime.now(UTC).isoformat(),
        'analysis_type': (
            'retrospective diagnostic using frozen open candidates'
        ),
        'sealed_boundary': {
            'double_holdout_opened': False,
            'prohibited_path_component': 'double_holdout',
        },
        'candidate_rule': (
            'm2_effective_dry from the frozen susceptibility and causal DWD '
            'threshold rules; selection does not use runoff outcome'
        ),
        'primary_confirmation_subset': (
            'spatial_test plus temporal_test effective-dry events (438 events)'
        ),
        'severe_stress_test_rule': (
            'direct_runoff_lh_mm >= within-basin open eligible-event q90; '
            'outcome-selected and therefore descriptive only'
        ),
        'named_cases_are_descriptive_only': True,
        'control_rule': {
            'same_basin': True,
            'same_season': True,
            'same_split': True,
            'control_dry_model': False,
            'control_classes': {
                'intermediate': 'train federal_state x season P35 to P65',
                'high_wetness': 'train federal_state x season >= P80',
            },
            'minimum_control_minus_candidate_dwd_0_30_nfk': (
                MINIMUM_CONTROL_WETNESS_GAP_NFK
            ),
            'matching_with_replacement': True,
            'ratio_bounds': {
                key: list(value) for key, value in MATCH_RATIO_BOUNDS.items()
            },
            'selection': (
                'minimum RMS log ratio over total rain, rain duration, and '
                'maximum 1 h, 3 h, and 6 h rainfall; event_id breaks ties'
            ),
        },
        'primary_outcome': (
            'effective-dry candidate minus intermediate-state control '
            'direct-runoff ratio'
        ),
        'robustness_outcomes': [
            'Lyne-Hollick direct runoff',
            'Eckhardt direct runoff',
            'event diagnostic CN at lambda 0.05',
            'raw-flow peak timing',
            'B1 residual at Lyne-Hollick and lambda 0.05',
        ],
        'limitations': [
            'controls are retrospective and not randomized',
            'the severe top-decile stress test is selected using '
            'Lyne-Hollick runoff',
            'DWD percent-nFK is a basin aggregate, not direct infiltration '
            'or repellency',
            'diagnostic event CN is inverted from runoff and is not ground '
            'truth',
        ],
        'inputs': {
            'open_event_catalog': verified_open,
            'predictions': source_record(args.predictions),
            'dwd_features': source_record(args.dwd_features),
            'hourly_source_root': str(args.timeseries_root.resolve()),
        },
        'model_frame_audit': model_audit,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / 'contract.json', contract)
    candidates.to_parquet(
        args.output_root / 'candidate_events.parquet',
        index=False,
        compression='zstd',
    )
    enriched.to_parquet(
        args.output_root / 'matched_event_features.parquet',
        index=False,
        compression='zstd',
    )
    pairs.to_parquet(
        args.output_root / 'matched_pairs.parquet',
        index=False,
        compression='zstd',
    )
    write_json(args.output_root / 'summary.json', summary)
    write_json(
        args.output_root / 'hourly_source_audit.json',
        {
            'schema': 'dynamic-cn-hourly-source-audit-v2',
            'source_count': len(hourly_sources),
            'sources': hourly_sources,
            'catalog_rain_and_peak_flow_values_reproduced': True,
        },
    )
    outputs = []
    for path in sorted(args.output_root.iterdir()):
        if path.name == 'artifact_manifest.json' or not path.is_file():
            continue
        outputs.append(
            {
                'path': path.name,
                'bytes': path.stat().st_size,
                'sha256': sha256(path),
            }
        )
    write_json(
        args.output_root / 'artifact_manifest.json',
        {
            'schema': 'dynamic-cn-open-event-confirmation-manifest-v2',
            'files': outputs,
        },
    )


if __name__ == '__main__':
    main()
