"""Tests for the preregistered dynamic-CN event catalogue builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'soil_state'
    / 'build_dynamic_cn_event_catalog.py'
)
SPEC = importlib.util.spec_from_file_location(
    'build_dynamic_cn_event_catalog', SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_delineation_splits_only_after_twelve_dry_hours() -> None:
    index = pd.date_range('2020-01-01', periods=30, freq='h')
    rain = np.zeros(30)
    rain[[0, 12, 25]] = 1.0
    events = MODULE.delineate_rain_events(index, rain)
    assert events == [(0, 12), (25, 25)]


def test_baseflow_filters_are_bounded_and_preserve_missing_gap() -> None:
    flow = np.array([1.0, 1.0, 3.0, 2.0, np.nan, 1.0, 1.0])
    for baseflow in (
        MODULE.lyne_hollick_baseflow(flow),
        MODULE.eckhardt_baseflow(flow),
    ):
        finite = np.isfinite(flow)
        assert np.isnan(baseflow[4])
        assert np.all(baseflow[finite] >= 0.0)
        assert np.all(baseflow[finite] <= flow[finite])


def test_scs_cn_diagnostic_round_trip() -> None:
    for abstraction_ratio in (0.05, 0.20):
        runoff = MODULE.scs_runoff_mm(80.0, 72.0, abstraction_ratio)
        inverted = MODULE.invert_event_cn(80.0, runoff, abstraction_ratio)
        assert abs(inverted - 72.0) < 1e-6


def test_event_extraction_retains_zero_runoff_and_never_uses_small_storm() -> None:
    index = pd.date_range('2020-06-01', periods=1000, freq='h')
    rain = np.zeros(len(index))
    rain[800:805] = 5.0
    rain[900:902] = 2.0
    frame = pd.DataFrame(
        {
            'date': index,
            'precipitation_mean_gapfilled': rain,
            'discharge_spec_obs': np.ones(len(index)),
            'air_temperature_mean': np.full(len(index), 15.0),
        }
    )
    config = {
        'wet_threshold': 0.1,
        'inter_event_dry_hours': 12,
        'minimum_event_rain': 10.0,
        'primary_event_rain': 20.0,
        'maximum_response_hours': 96,
        'minimum_coverage': 0.95,
        'lh_alpha': 0.925,
        'lh_passes': 3,
        'eckhardt_alpha': 0.98,
        'eckhardt_bfi_max': 0.80,
    }
    events, audit = MODULE.extract_events_from_frame(
        frame,
        basin='camelsde1h_DE000001',
        spatial_holdout=set(),
        strong_regulation=False,
        config=config,
    )
    assert len(events) == 1
    assert events.iloc[0]['event_rain_mm'] == 25.0
    assert events.iloc[0]['direct_runoff_lh_mm'] == 0.0
    assert np.isnan(events.iloc[0]['cn_event_lambda005_lh'])
    assert audit['below_sensitivity_rain_count'] == 1


def test_dwd_join_applies_strict_48_hour_lag(tmp_path: Path) -> None:
    dwd = tmp_path / 'dwd.npz'
    np.savez_compressed(
        dwd,
        basin=np.array(['camelsde1h_DE000001']),
        date=np.array(
            ['2020-01-01', '2020-01-02', '2020-01-03'],
            dtype='datetime64[D]',
        ),
        dwd_soil_0_30_p50=np.array([[10.0, 20.0, 99.0]], dtype=np.float32),
    )
    events = pd.DataFrame(
        {
            'basin': ['camelsde1h_DE000001'],
            'event_start': [pd.Timestamp('2020-01-03 18:00')],
            'analysis_eligible_both_pre_dwd': [True],
            'primary_rain_threshold': [True],
        }
    )
    joined = MODULE.join_dwd_features(events, dwd)
    assert joined.iloc[0]['dwd_state_valid_date'] == pd.Timestamp('2020-01-01')
    assert joined.iloc[0]['dwd_soil_0_30_p50'] == 10.0
    assert joined.iloc[0]['dwd_history_days'] == 1
