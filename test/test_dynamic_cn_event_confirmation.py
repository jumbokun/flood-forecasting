"""Tests for retrospective open-split dry-soil event confirmation."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).parents[1] / 'scripts' / 'soil_state'
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / 'confirm_dynamic_cn_events.py'
SPEC = importlib.util.spec_from_file_location(
    'confirm_dynamic_cn_events', SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reject_sealed_path_is_component_specific() -> None:
    with pytest.raises(RuntimeError, match='sealed double-holdout'):
        MODULE.reject_sealed_path(
            Path('/data/project/double_holdout/events.parquet')
        )
    MODULE.reject_sealed_path(Path('/data/project/double_holdout_note.json'))


def test_rolling_max_sum_requires_complete_window() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    assert MODULE.rolling_max_sum(values, 3) == 15.0
    assert MODULE.rolling_max_sum(np.array([1.0, 2.0]), 3) == 3.0


def test_json_compatible_replaces_nonfinite_values() -> None:
    assert MODULE.json_compatible({'x': [np.float64(np.nan), math.inf]}) == {
        'x': [None, None]
    }


def test_preliminary_controls_are_wet_same_basin_season_and_split() -> None:
    frame = pd.DataFrame(
        {
            'event_id': ['dry', 'good', 'wrong_season', 'dry_control'],
            'basin': ['a'] * 4,
            'season': ['JJA', 'JJA', 'SON', 'JJA'],
            'split': ['train', 'train', 'train', 'validation'],
            'dry_model': [True, False, False, True],
            'control_intermediate': [False, True, True, False],
            'dwd_soil_0_30_p50': [20.0, 40.0, 40.0, 20.0],
            'event_rain_mm': [30.0, 32.0, 30.0, 30.0],
            'rain_hours': [10, 11, 10, 10],
        }
    )
    pool = MODULE.preliminary_control_pool(
        frame, frame.iloc[[0]], 'intermediate'
    )
    assert pool['event_id'].tolist() == ['good']
    assert pool['candidate_event_id'].tolist() == ['dry']


def test_control_state_classes_use_training_quantiles_only() -> None:
    frame = pd.DataFrame(
        {
            'federal_state': ['x'] * 6,
            'season': ['JJA'] * 6,
            'split': ['train'] * 5 + ['spatial_test'],
            'dwd_soil_0_30_p50': [10.0, 30.0, 50.0, 70.0, 90.0, 55.0],
            'dry_model': [True, False, False, False, False, False],
        }
    )
    classified = MODULE.add_control_state_classes(frame)
    assert bool(classified.iloc[-1]['control_intermediate']) is True
    assert bool(classified.iloc[-1]['control_high_wetness']) is False


def test_select_matches_uses_rainfall_shape_and_deterministic_tie_break() -> (
    None
):
    candidates = pd.DataFrame(
        {
            'event_id': ['dry'],
            'event_rain_mm': [30.0],
            'rain_hours': [10.0],
        }
    )
    pool = pd.DataFrame(
        {
            'candidate_event_id': ['dry', 'dry'],
            'event_id': ['wet_b', 'wet_a'],
            'event_rain_mm': [30.0, 30.0],
            'rain_hours': [10.0, 10.0],
        }
    )
    hourly = pd.DataFrame(
        {
            'event_id': ['dry', 'wet_a', 'wet_b'],
            'rain_max_1h_mm': [5.0, 5.0, 5.0],
            'rain_max_3h_mm': [10.0, 10.0, 10.0],
            'rain_max_6h_mm': [20.0, 20.0, 20.0],
        }
    )
    matches = MODULE.select_matches(candidates, pool, hourly)
    assert matches.iloc[0]['match_status'] == 'MATCHED'
    assert matches.iloc[0]['control_event_id'] == 'wet_a'
    assert matches.iloc[0]['match_log_rms_distance'] == 0.0
