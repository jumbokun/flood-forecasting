"""Tests for paired soil-state event diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'soil_state'
    / 'analyze_soil_state_events.py'
)
SPEC = importlib.util.spec_from_file_location(
    'analyze_soil_state_events', SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def example_panels() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    lead = np.arange(49)
    issue = np.array([['2024-01-01']], dtype='datetime64[ns]')
    obs = np.arange(49, dtype=np.float32).reshape(1, 1, 49) / 100
    c0 = {
        'basin': np.array(['basin_a']),
        'issue_time': issue,
        'lead_hour': lead,
        'obs_mm_h': obs,
        'sim_mm_h': obs + 0.2,
    }
    s1 = {
        **c0,
        'sim_mm_h': obs + 0.1,
    }
    return c0, s1


def test_event_error_gain_is_positive_when_s1_is_closer() -> None:
    c0, s1 = example_panels()
    MODULE.validate_panels(c0, s1)
    frame, audit = MODULE.event_error_frame(c0, s1)

    assert len(frame) == 1
    assert audit['lead_window_hours'] == [25, 48]
    assert frame.iloc[0]['valid_hours'] == 24
    assert frame.iloc[0]['mse_gain'] > 0
    assert frame.iloc[0]['mae_gain'] > 0
    assert frame.iloc[0]['peak_abs_error_gain'] > 0


def test_soil_and_rain_boundaries() -> None:
    soil = MODULE.classify_raw_soil(pd.Series([29.9, 30, 50, 100]))
    assert soil.tolist() == [
        'very_dry_lt30',
        'dry_30_50',
        'middle_50_100',
        'wet_ge100',
    ]
    rain = MODULE.classify_absolute_rain(pd.Series([0, 1, 10, 30]))
    assert rain.tolist() == ['lt1mm', '1_10mm', '10_30mm', 'ge30mm']


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, 0.2])
    adjusted = MODULE.benjamini_hochberg(p_values)
    ordered = adjusted[np.argsort(p_values.to_numpy())]
    assert np.all(np.diff(ordered) >= 0)
    assert np.all((adjusted >= 0) & (adjusted <= 1))
