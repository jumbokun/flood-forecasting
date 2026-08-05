#!/usr/bin/env python3
"""Diagnose conditional DWD soil-state value on paired forecast events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binomtest

LEAD_START = 25
LEAD_END = 48
MIN_VALID_HOURS = 18
SOIL_QUINTILE_LABELS = [
    'q1_very_dry',
    'q2_dry',
    'q3_middle',
    'q4_wet',
    'q5_very_wet',
]
STRATIFIERS = [
    'soil_percentile_quintile',
    'soil_raw_class',
    'rain_percentile_class',
    'rain_absolute_class',
    'soil_x_rain',
    'season',
    'soil_texture',
    'landcover',
    'federal_state',
    'area_class',
]
GATE = {
    'all_events': {
        'minimum_basins': 40,
        'minimum_events': 500,
        'minimum_median_relative_mse_gain': 0.02,
        'minimum_fraction_basins_better': 0.55,
        'require_bootstrap_ci_low_above_zero': True,
        'require_median_relative_mae_gain_above_zero': True,
        'maximum_bh_q_value': 0.05,
    },
    'flood_events': {
        'minimum_basins': 20,
        'minimum_events': 100,
        'require_relative_mse_ci_low_above_zero': True,
        'require_peak_absolute_error_gain_above_zero': True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--c0-panel', required=True, type=Path)
    parser.add_argument('--s1-panel', required=True, type=Path)
    parser.add_argument('--dwd-daily', required=True, type=Path)
    parser.add_argument('--radklim-zarr', required=True, type=Path)
    parser.add_argument('--soil-attributes', required=True, type=Path)
    parser.add_argument('--landcover-attributes', required=True, type=Path)
    parser.add_argument('--topographic-attributes', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--availability-lag-hours', type=int, default=48)
    parser.add_argument('--bootstrap-replicates', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260805)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def validate_panels(
    c0: dict[str, np.ndarray], s1: dict[str, np.ndarray]
) -> None:
    """Require a strictly paired C0/S1 issue-time experiment."""

    for key in ('basin', 'issue_time', 'lead_hour'):
        if not np.array_equal(c0[key], s1[key]):
            raise RuntimeError(f'C0/S1 panel mismatch in {key}.')
    if c0['obs_mm_h'].shape != s1['obs_mm_h'].shape:
        raise RuntimeError('C0/S1 observation shapes differ.')
    if not np.allclose(
        c0['obs_mm_h'], s1['obs_mm_h'], equal_nan=True, atol=1e-7
    ):
        raise RuntimeError('C0/S1 observations are not identical.')
    expected = np.arange(0, 49)
    if not np.array_equal(c0['lead_hour'], expected):
        raise RuntimeError(
            f'Expected lead hours 0..48, found {c0["lead_hour"]}.'
        )


def season_from_month(month: pd.Series) -> pd.Series:
    result = pd.Series(index=month.index, dtype='object')
    result[month.isin([12, 1, 2])] = 'winter'
    result[month.isin([3, 4, 5])] = 'spring'
    result[month.isin([6, 7, 8])] = 'summer'
    result[month.isin([9, 10, 11])] = 'autumn'
    return result


def classify_raw_soil(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 30, 50, 100, np.inf],
        labels=['very_dry_lt30', 'dry_30_50', 'middle_50_100', 'wet_ge100'],
        right=False,
    ).astype('object')


def classify_soil_percentile(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=SOIL_QUINTILE_LABELS,
        right=True,
    ).astype('object')


def classify_rain_percentile(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.5, 0.9, np.inf],
        labels=['lower_50', 'p50_90', 'top_10'],
        right=True,
    ).astype('object')


def classify_absolute_rain(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 1, 10, 30, np.inf],
        labels=['lt1mm', '1_10mm', '10_30mm', 'ge30mm'],
        right=False,
    ).astype('object')


def event_error_frame(
    c0: dict[str, np.ndarray], s1: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict]:
    lead = c0['lead_hour']
    keep_lead = (lead >= LEAD_START) & (lead <= LEAD_END)
    obs = c0['obs_mm_h'][:, :, keep_lead].astype(np.float64)
    c0_sim = c0['sim_mm_h'][:, :, keep_lead].astype(np.float64)
    s1_sim = s1['sim_mm_h'][:, :, keep_lead].astype(np.float64)
    valid = np.isfinite(obs) & np.isfinite(c0_sim) & np.isfinite(s1_sim)
    valid_hours = valid.sum(axis=2)

    c0_error = np.where(valid, c0_sim - obs, np.nan)
    s1_error = np.where(valid, s1_sim - obs, np.nan)
    c0_sse = np.nansum(c0_error**2, axis=2)
    s1_sse = np.nansum(s1_error**2, axis=2)
    c0_sae = np.nansum(np.abs(c0_error), axis=2)
    s1_sae = np.nansum(np.abs(s1_error), axis=2)
    denominator = np.maximum(valid_hours, 1)

    masked_obs = np.where(valid, obs, -np.inf)
    masked_c0 = np.where(valid, c0_sim, -np.inf)
    masked_s1 = np.where(valid, s1_sim, -np.inf)
    obs_peak = masked_obs.max(axis=2)
    c0_peak = masked_c0.max(axis=2)
    s1_peak = masked_s1.max(axis=2)
    incomplete = valid_hours < MIN_VALID_HOURS
    for values in (obs_peak, c0_peak, s1_peak):
        values[incomplete] = np.nan

    basin = np.repeat(c0['basin'], c0['issue_time'].shape[1])
    issue_time = c0['issue_time'].reshape(-1)
    frame = pd.DataFrame(
        {
            'basin': basin,
            'issue_time': pd.to_datetime(issue_time),
            'valid_hours': valid_hours.reshape(-1),
            'c0_sse': c0_sse.reshape(-1),
            's1_sse': s1_sse.reshape(-1),
            'c0_sae': c0_sae.reshape(-1),
            's1_sae': s1_sae.reshape(-1),
            'c0_mse': (c0_sse / denominator).reshape(-1),
            's1_mse': (s1_sse / denominator).reshape(-1),
            'c0_mae': (c0_sae / denominator).reshape(-1),
            's1_mae': (s1_sae / denominator).reshape(-1),
            'obs_peak_mm_h': obs_peak.reshape(-1),
            'c0_peak_mm_h': c0_peak.reshape(-1),
            's1_peak_mm_h': s1_peak.reshape(-1),
        }
    )
    frame['mse_gain'] = frame['c0_mse'] - frame['s1_mse']
    frame['mae_gain'] = frame['c0_mae'] - frame['s1_mae']
    frame['peak_abs_error_gain'] = (
        frame['c0_peak_mm_h'] - frame['obs_peak_mm_h']
    ).abs() - (frame['s1_peak_mm_h'] - frame['obs_peak_mm_h']).abs()
    total = len(frame)
    frame = (
        frame[frame['valid_hours'] >= MIN_VALID_HOURS]
        .copy()
        .reset_index(drop=True)
    )
    return frame, {
        'total_basin_issue_windows': total,
        'valid_basin_issue_windows': len(frame),
        'minimum_valid_hours': MIN_VALID_HOURS,
        'lead_window_hours': [LEAD_START, LEAD_END],
    }


def attach_dwd_state(
    frame: pd.DataFrame,
    *,
    dwd_path: Path,
    availability_lag_hours: int,
) -> dict:
    with np.load(dwd_path, allow_pickle=False) as payload:
        dwd_basins = payload['basin'].astype(str)
        dwd_dates = payload['date'].astype('datetime64[D]')
        soil_010 = payload['dwd_soil_0_10_p50'].astype(np.float32)
        soil_030 = payload['dwd_soil_0_30_p50'].astype(np.float32)
        coverage_030 = payload['dwd_soil_0_30_coverage'].astype(np.float32)

    basin_lookup = {value: index for index, value in enumerate(dwd_basins)}
    date_lookup = {value: index for index, value in enumerate(dwd_dates)}
    source_date = (
        frame['issue_time'].to_numpy(dtype='datetime64[h]')
        - np.timedelta64(availability_lag_hours, 'h')
    ).astype('datetime64[D]')
    basin_index = np.array(
        [basin_lookup[value] for value in frame['basin']], dtype=np.int64
    )
    date_index = np.array(
        [date_lookup[value] for value in source_date], dtype=np.int64
    )
    frame['dwd_source_date'] = source_date
    frame['dwd_soil_0_10_p50'] = soil_010[basin_index, date_index]
    frame['dwd_soil_0_30_p50'] = soil_030[basin_index, date_index]
    frame['dwd_soil_0_30_coverage'] = coverage_030[basin_index, date_index]

    reference_dates = dwd_dates < np.datetime64('2024-01-01')
    percentiles = np.full(len(frame), np.nan, dtype=np.float64)
    for basin, positions in frame.groupby('basin', sort=False).groups.items():
        bidx = basin_lookup[basin]
        reference = soil_030[bidx, reference_dates]
        reference = np.sort(reference[np.isfinite(reference)])
        values = frame.loc[positions, 'dwd_soil_0_30_p50'].to_numpy()
        if len(reference):
            percentiles[np.asarray(positions)] = np.searchsorted(
                reference, values, side='right'
            ) / len(reference)
    frame['soil_historical_percentile'] = percentiles
    return {
        'reference_period_end_exclusive': '2024-01-01',
        'availability_lag_hours': availability_lag_hours,
        'soil_state_variable': 'dwd_soil_0_30_p50',
    }


def attach_realized_rain(frame: pd.DataFrame, *, radklim_path: Path) -> dict:
    group = zarr.open_group(str(radklim_path), mode='r')
    basin_values = group['basin'][:].astype(str)
    hours = group['date'][:].astype(np.int64)
    units = group['date'].attrs['units']
    prefix = 'hours since '
    if not units.startswith(prefix):
        raise RuntimeError(f'Unsupported RADKLIM date units: {units}')
    origin = np.datetime64(units.removeprefix(prefix).replace(' ', 'T'), 'h')
    dates = origin + hours.astype('timedelta64[h]')
    basin_lookup = {value: index for index, value in enumerate(basin_values)}
    date_lookup = {value: index for index, value in enumerate(dates)}
    precipitation = group['radklim_total_precipitation']

    rain_024 = np.full(len(frame), np.nan, dtype=np.float32)
    rain_2548 = np.full(len(frame), np.nan, dtype=np.float32)
    rain_048 = np.full(len(frame), np.nan, dtype=np.float32)
    rain_max_1h = np.full(len(frame), np.nan, dtype=np.float32)
    for basin, positions in frame.groupby('basin', sort=False).groups.items():
        positions = np.asarray(positions)
        bidx = basin_lookup[basin]
        row = np.asarray(precipitation[bidx, :], dtype=np.float32)
        issue_values = frame.loc[positions, 'issue_time'].to_numpy(
            dtype='datetime64[h]'
        )
        issue_indexes = np.array(
            [date_lookup[value] for value in issue_values], dtype=np.int64
        )
        indexes = issue_indexes[:, None] + np.arange(1, 49)[None, :]
        values = row[indexes]
        finite_count = np.isfinite(values).sum(axis=1)
        acceptable = finite_count >= 44
        rain_024[positions] = np.where(
            acceptable, np.nansum(values[:, :24], axis=1), np.nan
        )
        rain_2548[positions] = np.where(
            acceptable, np.nansum(values[:, 24:], axis=1), np.nan
        )
        rain_048[positions] = np.where(
            acceptable, np.nansum(values, axis=1), np.nan
        )
        rain_max_1h[positions] = np.where(
            acceptable, np.nanmax(values, axis=1), np.nan
        )
    frame['realized_rain_0_24_mm'] = rain_024
    frame['realized_rain_25_48_mm'] = rain_2548
    frame['realized_rain_0_48_mm'] = rain_048
    frame['realized_rain_max_1h_mm'] = rain_max_1h
    frame['rain_within_basin_percentile'] = frame.groupby('basin')[
        'realized_rain_25_48_mm'
    ].rank(method='average', pct=True)
    return {
        'variable': 'radklim_total_precipitation',
        'diagnostic_only': (
            'Realized future RADKLIM rainfall is used only to stratify '
            'outcomes; it is not an issue-time model input.'
        ),
        'minimum_finite_hours_out_of_48': 44,
    }


def dominant_label(frame: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    values = frame[list(columns)].rename(columns=columns)
    return values.idxmax(axis=1)


def attach_static_attributes(
    frame: pd.DataFrame,
    *,
    soil_path: Path,
    landcover_path: Path,
    topographic_path: Path,
) -> None:
    soil = pd.read_csv(soil_path)
    landcover = pd.read_csv(landcover_path)
    topographic = pd.read_csv(topographic_path)
    static = soil.merge(landcover, on='gauge_id', validate='one_to_one').merge(
        topographic[
            [
                'gauge_id',
                'federal_state',
                'area',
                'gauge_name',
                'water_body_name',
            ]
        ],
        on='gauge_id',
        validate='one_to_one',
    )
    static['soil_texture'] = dominant_label(
        static,
        {
            'clay_0_30cm_mean': 'clay_dominant',
            'silt_0_30cm_mean': 'silt_dominant',
            'sand_0_30cm_mean': 'sand_dominant',
        },
    )
    static['landcover'] = dominant_label(
        static,
        {
            'artificial_surfaces_perc': 'artificial_dominant',
            'agricultural_areas_perc': 'agriculture_dominant',
            'forests_and_seminatural_areas_perc': 'forest_dominant',
            'wetlands_perc': 'wetland_dominant',
            'water_bodies_perc': 'water_dominant',
        },
    )
    static['area_class'] = pd.qcut(
        static['area'],
        q=3,
        labels=['small', 'medium', 'large'],
        duplicates='drop',
    ).astype('object')
    static['basin'] = 'camelsde1h_' + static['gauge_id'].astype(str)
    columns = [
        'basin',
        'gauge_name',
        'water_body_name',
        'federal_state',
        'area',
        'area_class',
        'soil_texture',
        'landcover',
        'clay_0_30cm_mean',
        'silt_0_30cm_mean',
        'sand_0_30cm_mean',
        'artificial_surfaces_perc',
        'agricultural_areas_perc',
        'forests_and_seminatural_areas_perc',
    ]
    joined = frame.merge(
        static[columns], on='basin', how='left', validate='many_to_one'
    )
    if joined['federal_state'].isna().any():
        missing = joined.loc[joined['federal_state'].isna(), 'basin'].unique()
        raise RuntimeError(f'Missing static attributes for {missing}.')
    for column in joined.columns:
        frame[column] = joined[column].to_numpy()


def add_classes(frame: pd.DataFrame) -> None:
    frame['soil_raw_class'] = classify_raw_soil(frame['dwd_soil_0_30_p50'])
    frame['soil_percentile_quintile'] = classify_soil_percentile(
        frame['soil_historical_percentile']
    )
    frame['rain_percentile_class'] = classify_rain_percentile(
        frame['rain_within_basin_percentile']
    )
    frame['rain_absolute_class'] = classify_absolute_rain(
        frame['realized_rain_25_48_mm']
    )
    frame['soil_x_rain'] = pd.Series(index=frame.index, dtype='object')
    cross_valid = (
        frame['soil_percentile_quintile'].notna()
        & frame['rain_percentile_class'].notna()
    )
    frame.loc[cross_valid, 'soil_x_rain'] = (
        frame.loc[cross_valid, 'soil_percentile_quintile'].astype(str)
        + '|'
        + frame.loc[cross_valid, 'rain_percentile_class'].astype(str)
    )
    event_midpoint = frame['issue_time'] + pd.Timedelta(hours=36)
    frame['season'] = season_from_month(event_midpoint.dt.month)
    thresholds = frame.groupby('basin')['obs_peak_mm_h'].transform(
        lambda values: values.quantile(0.9)
    )
    frame['is_flood_event'] = frame['obs_peak_mm_h'] >= thresholds


def basin_group_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    c0_sse = group['c0_sse'].sum()
    s1_sse = group['s1_sse'].sum()
    c0_sae = group['c0_sae'].sum()
    s1_sae = group['s1_sae'].sum()
    return {
        'n_events': len(group),
        'relative_mse_gain': 1 - s1_sse / c0_sse if c0_sse > 0 else np.nan,
        'relative_mae_gain': 1 - s1_sae / c0_sae if c0_sae > 0 else np.nan,
        'median_peak_absolute_error_gain': group[
            'peak_abs_error_gain'
        ].median(),
        'fraction_events_s1_lower_mse': (group['mse_gain'] > 0).mean(),
    }


def bootstrap_median(
    values: np.ndarray, *, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    sampled = rng.choice(values, size=(replicates, len(values)), replace=True)
    medians = np.median(sampled, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def summarize_strata(
    frame: pd.DataFrame,
    *,
    subset: str,
    replicates: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for stratifier in STRATIFIERS:
        selected = frame.dropna(subset=[stratifier])
        for stratum, group in selected.groupby(stratifier, observed=True):
            basin_rows = []
            for basin, basin_group in group.groupby('basin'):
                basin_rows.append(
                    {'basin': basin, **basin_group_metrics(basin_group)}
                )
            basin_frame = pd.DataFrame(basin_rows).dropna(
                subset=['relative_mse_gain']
            )
            if basin_frame.empty:
                continue
            values = basin_frame['relative_mse_gain'].to_numpy()
            low, high = bootstrap_median(values, replicates=replicates, rng=rng)
            nonzero = values[values != 0]
            sign_p = (
                float(
                    binomtest(
                        int((nonzero > 0).sum()),
                        len(nonzero),
                        p=0.5,
                        alternative='greater',
                    ).pvalue
                )
                if len(nonzero)
                else 1.0
            )
            rows.append(
                {
                    'subset': subset,
                    'stratifier': stratifier,
                    'stratum': str(stratum),
                    'n_events': len(group),
                    'n_basins': len(basin_frame),
                    'median_relative_mse_gain': float(np.median(values)),
                    'bootstrap_median_ci_low': low,
                    'bootstrap_median_ci_high': high,
                    'median_relative_mae_gain': float(
                        basin_frame['relative_mae_gain'].median()
                    ),
                    'median_peak_absolute_error_gain': float(
                        basin_frame['median_peak_absolute_error_gain'].median()
                    ),
                    'fraction_basins_s1_better_mse': float((values > 0).mean()),
                    'median_fraction_events_s1_lower_mse': float(
                        basin_frame['fraction_events_s1_lower_mse'].median()
                    ),
                    'sign_test_p_value': sign_p,
                }
            )
    result = pd.DataFrame(rows)
    result['bh_q_value'] = benjamini_hochberg(result['sign_test_p_value'])
    return result


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def apply_gate(all_events: pd.DataFrame, floods: pd.DataFrame) -> pd.DataFrame:
    gate = GATE['all_events']
    candidates = all_events[
        (all_events['n_basins'] >= gate['minimum_basins'])
        & (all_events['n_events'] >= gate['minimum_events'])
        & (
            all_events['median_relative_mse_gain']
            >= gate['minimum_median_relative_mse_gain']
        )
        & (all_events['bootstrap_median_ci_low'] > 0)
        & (all_events['median_relative_mae_gain'] > 0)
        & (
            all_events['fraction_basins_s1_better_mse']
            >= gate['minimum_fraction_basins_better']
        )
        & (all_events['bh_q_value'] <= gate['maximum_bh_q_value'])
    ].copy()
    floods = floods.add_prefix('flood_').rename(
        columns={
            'flood_stratifier': 'stratifier',
            'flood_stratum': 'stratum',
        }
    )
    candidates = candidates.merge(
        floods, on=['stratifier', 'stratum'], how='left', validate='one_to_one'
    )
    flood_gate = GATE['flood_events']
    candidates['passes_flood_confirmation'] = (
        (candidates['flood_n_basins'] >= flood_gate['minimum_basins'])
        & (candidates['flood_n_events'] >= flood_gate['minimum_events'])
        & (candidates['flood_bootstrap_median_ci_low'] > 0)
        & (candidates['flood_median_peak_absolute_error_gain'] > 0)
    )
    return candidates[candidates['passes_flood_confirmation']].copy()


def plot_soil_rain_heatmap(summary: pd.DataFrame, output: Path) -> None:
    selected = summary[summary['stratifier'] == 'soil_x_rain'].copy()
    if selected.empty:
        return
    split = selected['stratum'].str.split('|', expand=True, regex=False)
    selected['soil'] = split[0]
    selected['rain'] = split[1]
    pivot = selected.pivot(
        index='soil', columns='rain', values='median_relative_mse_gain'
    ).reindex(
        index=SOIL_QUINTILE_LABELS,
        columns=['lower_50', 'p50_90', 'top_10'],
    )
    values = pivot.to_numpy(dtype=float)
    maximum = max(0.01, float(np.nanmax(np.abs(values))))
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    image = axis.imshow(
        values,
        cmap='RdBu',
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
        aspect='auto',
    )
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_xlabel('Within-basin realized 25–48 h rainfall class')
    axis.set_ylabel('Historical DWD 0–30 cm state quintile')
    axis.set_title('Median basin-balanced relative MSE gain (S1 over C0)')
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                axis.text(
                    column,
                    row,
                    f'{100 * values[row, column]:+.1f}%',
                    ha='center',
                    va='center',
                    fontsize=9,
                )
    figure.colorbar(image, ax=axis, label='Relative MSE gain')
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    c0 = load_panel(args.c0_panel)
    s1 = load_panel(args.s1_panel)
    validate_panels(c0, s1)
    frame, event_audit = event_error_frame(c0, s1)
    dwd_contract = attach_dwd_state(
        frame,
        dwd_path=args.dwd_daily,
        availability_lag_hours=args.availability_lag_hours,
    )
    rain_contract = attach_realized_rain(frame, radklim_path=args.radklim_zarr)
    attach_static_attributes(
        frame,
        soil_path=args.soil_attributes,
        landcover_path=args.landcover_attributes,
        topographic_path=args.topographic_attributes,
    )
    add_classes(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / 'event_panel.csv.gz', index=False, compression='gzip'
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    all_summary = summarize_strata(
        frame,
        subset='all_events',
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    flood_summary = summarize_strata(
        frame[frame['is_flood_event']],
        subset='flood_events',
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    all_summary.to_csv(args.output_dir / 'strata_all_events.csv', index=False)
    flood_summary.to_csv(
        args.output_dir / 'strata_flood_events.csv', index=False
    )
    candidates = apply_gate(all_summary, flood_summary)
    candidates.to_csv(args.output_dir / 'candidate_signals.csv', index=False)
    plot_soil_rain_heatmap(
        all_summary, args.output_dir / 'soil_rain_mse_gain_heatmap.png'
    )

    payload = {
        'schema_version': 1,
        'decision': (
            'TARGETED_ABLATION_WARRANTED_NOT_FULL_TRAINING'
            if len(candidates)
            else 'NO_CONDITIONAL_SIGNAL_AT_PREDECLARED_GATE'
        ),
        'interpretation_boundary': (
            'This single-seed post-hoc diagnostic can prioritize targeted '
            'ablations. It cannot authorize production or full national '
            'training.'
        ),
        'event_contract': event_audit,
        'dwd_contract': dwd_contract,
        'rain_contract': rain_contract,
        'gate': GATE,
        'bootstrap_replicates': args.bootstrap_replicates,
        'bootstrap_seed': args.bootstrap_seed,
        'n_test_basins': int(frame['basin'].nunique()),
        'n_valid_events': len(frame),
        'n_flood_events': int(frame['is_flood_event'].sum()),
        'candidate_signal_count': len(candidates),
        'sources': {
            'c0_panel': {
                'path': str(args.c0_panel),
                'sha256': sha256(args.c0_panel),
            },
            's1_panel': {
                'path': str(args.s1_panel),
                'sha256': sha256(args.s1_panel),
            },
            'dwd_daily': {
                'path': str(args.dwd_daily),
                'sha256': sha256(args.dwd_daily),
            },
        },
    }
    (args.output_dir / 'conclusion.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
