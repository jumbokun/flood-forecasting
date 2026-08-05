"""Tests for the issue-time prediction panel contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'soil_state'
    / 'evaluate_issue_time_forecast.py'
)
SPEC = importlib.util.spec_from_file_location(
    'evaluate_issue_time_forecast', SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_prediction_panel_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / 'model.pt'
    config = tmp_path / 'config.yml'
    checkpoint.write_bytes(b'model')
    config.write_text('config\n', encoding='utf-8')
    issue_time = np.array(['2024-01-01', '2024-01-02'], dtype='datetime64[ns]')
    lead_hour = np.array([0, 1, 2])
    panels = []
    for basin_index in range(2):
        obs = np.arange(6, dtype=np.float32).reshape(2, 3) + basin_index
        panels.append(
            {
                'basin': f'basin_{basin_index}',
                'issue_time': issue_time,
                'lead_hour': lead_hour,
                'obs_mm_h': obs,
                'sim_mm_h': obs + 0.5,
            }
        )

    output = tmp_path / 'panel.npz'
    audit = MODULE.write_prediction_panel(
        output,
        panels,
        checkpoint=checkpoint,
        config_path=config,
        issue_stride_hours=24,
    )

    assert audit['shape'] == {'basin': 2, 'issue_time': 2, 'lead_hour': 3}
    with np.load(output, allow_pickle=False) as payload:
        assert payload['obs_mm_h'].shape == (2, 2, 3)
        assert payload['sim_mm_h'].dtype == np.float32
        assert payload['basin'].tolist() == ['basin_0', 'basin_1']
        assert payload['issue_stride_hours'].item() == 24


def test_prediction_panel_rejects_mismatched_issue_counts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / 'model.pt'
    config = tmp_path / 'config.yml'
    checkpoint.write_bytes(b'model')
    config.write_text('config\n', encoding='utf-8')
    panels = [
        {
            'basin': 'a',
            'issue_time': np.array(['2024-01-01'], dtype='datetime64[ns]'),
            'lead_hour': np.array([0, 1]),
            'obs_mm_h': np.ones((1, 2)),
            'sim_mm_h': np.ones((1, 2)),
        },
        {
            'basin': 'b',
            'issue_time': np.array(
                ['2024-01-01', '2024-01-02'], dtype='datetime64[ns]'
            ),
            'lead_hour': np.array([0, 1]),
            'obs_mm_h': np.ones((2, 2)),
            'sim_mm_h': np.ones((2, 2)),
        },
    ]

    try:
        MODULE.write_prediction_panel(
            tmp_path / 'panel.npz',
            panels,
            checkpoint=checkpoint,
            config_path=config,
            issue_stride_hours=24,
        )
    except RuntimeError as error:
        assert 'not uniform' in str(error)
    else:
        raise AssertionError('Expected mismatched issue counts to fail.')
