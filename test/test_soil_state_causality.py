from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'soil_state'
    / 'materialize_hourly_soil_product.py'
)
SPEC = importlib.util.spec_from_file_location(
    'materialize_hourly_soil_product', SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_conservative_48h_join_boundary():
    daily = np.array(
        ['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]'
    )
    issue = np.array(
        ['2024-01-02T23', '2024-01-03T00', '2024-01-03T23', '2024-01-04T00'],
        dtype='datetime64[h]',
    )
    indices, available = MODULE.causal_source_indices(daily, issue, 48)
    assert available.tolist() == [False, True, True, True]
    assert indices.tolist() == [0, 0, 0, 1]


def test_join_never_uses_state_after_cutoff():
    daily = np.arange(
        np.datetime64('2020-01-01'),
        np.datetime64('2020-02-01'),
        dtype='datetime64[D]',
    )
    issue = np.arange(
        np.datetime64('2020-01-03T00'),
        np.datetime64('2020-02-01T00'),
        np.timedelta64(1, 'h'),
    )
    indices, available = MODULE.causal_source_indices(daily, issue, 48)
    selected = daily[indices[available]]
    cutoff = (issue[available] - np.timedelta64(48, 'h')).astype(
        'datetime64[D]'
    )
    assert np.all(selected <= cutoff)
