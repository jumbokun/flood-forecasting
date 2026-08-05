"""Tests for the open-split interpretable dynamic-CN model runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'soil_state'
    / 'train_dynamic_cn_models.py'
)
SPEC = importlib.util.spec_from_file_location('train_dynamic_cn_models', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_susceptibility_mask_excludes_high_clay_and_wetland() -> None:
    frame = pd.DataFrame(
        {
            'sand_0_30cm_mean': [40.0, 40.0, 20.0],
            'soil_organic_carbon_0_30cm_mean': [60.0, 60.0, 20.0],
            'forests_and_seminatural_areas_perc': [50.0, 50.0, 20.0],
            'agricultural_areas_perc': [20.0, 20.0, 80.0],
            'silt_0_30cm_mean': [30.0, 30.0, 55.0],
            'clay_0_30cm_mean': [20.0, 35.0, 20.0],
            'wetlands_perc': [0.0, 0.0, 6.0],
            'water_bodies_perc': [0.0, 0.0, 0.0],
        }
    )
    mask = MODULE.susceptibility_mask(frame)
    assert mask['m2_susceptible'].tolist() == [True, False, False]
    assert mask['shrink_swell_clay_separate'].tolist() == [False, True, False]


def test_grouped_shuffle_is_deterministic_and_group_local() -> None:
    frame = pd.DataFrame(
        {
            'basin': ['a'] * 4 + ['b'] * 4,
            'season': ['JJA'] * 8,
            'split': ['train'] * 8,
            'x': np.arange(8, dtype=float),
            'y': np.arange(8, dtype=float) + 100,
        }
    )
    first = MODULE.grouped_shuffle(frame, ('x', 'y'))
    second = MODULE.grouped_shuffle(frame, ('x', 'y'))
    assert np.array_equal(first, second)
    assert set(first[:4, 0]) == set(frame.loc[:3, 'x'])
    assert set(first[4:, 0]) == set(frame.loc[4:, 'x'])
    assert np.all(first[:, 1] - first[:, 0] == 100)


def test_dynamic_cn_wet_and_dry_terms_are_monotone() -> None:
    model = MODULE.DynamicCN(1, 'M2')
    base = torch.zeros((3, 1))
    wet = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.5, 0.5]])
    dry = torch.tensor([0.0, 0.0, 0.5])
    cn = model(base, wet, dry).detach().numpy()
    assert cn[1] >= cn[0]
    assert cn[2] >= cn[1]


def test_basin_balanced_mae_uses_median_basin_error() -> None:
    basins = np.array(['a', 'a', 'b'])
    observed = np.zeros(3)
    predicted = np.array([2.0, 4.0, 10.0])
    assert MODULE.basin_balanced_mae(basins, observed, predicted) == 6.5


def test_benjamini_hochberg_is_rank_monotone() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, 0.20])
    adjusted = MODULE.benjamini_hochberg(p_values)
    ordered = adjusted[np.argsort(p_values)]
    assert np.all(np.diff(ordered) >= 0)
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))


def test_m2_sample_gate_stops_when_severe_events_are_insufficient() -> None:
    gate = MODULE.evaluate_m2_sample_gate(
        {
            'm2_effective_dry_event_count': 2627,
            'm2_effective_dry_basin_count': 131,
            'm2_effective_dry_top_decile_lh_event_count': 67,
        }
    )
    assert gate['passed'] is False
    assert gate['observed_top_decile_events'] == 67
