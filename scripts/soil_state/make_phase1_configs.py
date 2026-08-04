#!/usr/bin/env python3
"""Generate matched C0/C1/S1 soil-state Phase-1 experiment configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


MET_INPUTS = [
    'radklim_total_precipitation',
    'radklim_temperature_2m',
    'radklim_surface_net_solar_radiation',
    'time_doy_sin',
    'time_doy_cos',
]
FORECAST_INPUTS = [
    'iconfc_total_precipitation',
    'iconfc_temperature_2m',
]
SOIL_INPUTS = [
    'dwd_soil_0_10_p10',
    'dwd_soil_0_10_p50',
    'dwd_soil_0_10_p90',
    'dwd_soil_0_30_p10',
    'dwd_soil_0_30_p50',
    'dwd_soil_0_30_p90',
    'dwd_soil_0_10_coverage',
    'dwd_soil_0_30_coverage',
    'dwd_soil_layer_gradient_p50',
    'dwd_soil_state_age_hours',
]
GENERATED_KEYS = {
    'commit_hash',
    'img_log_dir',
    'number_of_basins',
    'package_version',
    'train_dir',
}


def write_config(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=1000), encoding='utf-8'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('/data/zhu/openhydronet/data/camels_de_1h_iconfc_soil_v1'),
    )
    parser.add_argument(
        '--run-root',
        type=Path,
        default=Path('/data/zhu/openhydronet/runs/soil-state-v1'),
    )
    parser.add_argument(
        '--precomputed-scaler-dir',
        type=Path,
        default=None,
        help='Directory containing scaler.nc shared by the matched experiments.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding='utf-8'))
    for key in GENERATED_KEYS:
        base.pop(key, None)
    data_root = args.data_root.resolve()
    base.update(
        {
            'data_dir': str(data_root),
            'dynamics_data_dir': str(data_root),
            'statics_data_dir': str(data_root),
            'targets_data_dir': str(data_root),
            'train_basin_file': str(data_root / 'basins/train_basins.txt'),
            'validation_basin_file': str(data_root / 'basins/val_basins.txt'),
            'test_basin_file': str(data_root / 'basins/test_basins.txt'),
            'run_dir': str(args.run_root.resolve()),
            'hindcast_inputs': {'met': MET_INPUTS},
            'forecast_inputs': {'iconfc': FORECAST_INPUTS},
            'precomputed_scaler_dir': (
                str(args.precomputed_scaler_dir.resolve())
                if args.precomputed_scaler_dir is not None
                else None
            ),
            'validate_every': None,
            'log_interval': 100,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        'c0': {
            'seq_length': 720,
            'forecast_overlap': 720,
            'hindcast_inputs': {'met': MET_INPUTS},
        },
        'c1': {
            'seq_length': 1440,
            'forecast_overlap': 1440,
            'hindcast_inputs': {'met': MET_INPUTS},
        },
        's1': {
            'seq_length': 720,
            'forecast_overlap': 720,
            'hindcast_inputs': {'met': MET_INPUTS, 'soil': SOIL_INPUTS},
        },
    }
    for name, override in experiments.items():
        for seed in (111, 222, 333):
            payload = deepcopy(base)
            payload.update(override)
            payload.update(
                {
                    'experiment_name': f'soil_state_{name}_screen_s{seed}',
                    'seed': seed,
                    'epochs': 3,
                    'max_updates_per_epoch': 500,
                    'save_weights_every': 3,
                }
            )
            write_config(
                args.output_dir / f'{name}_screen_s{seed}.yml', payload
            )

        confirm = deepcopy(base)
        confirm.update(override)
        confirm.update(
            {
                'experiment_name': f'soil_state_{name}_confirm_s111',
                'seed': 111,
                'epochs': 5,
                'max_updates_per_epoch': 2000,
                'save_weights_every': 5,
            }
        )
        write_config(args.output_dir / f'{name}_confirm_s111.yml', confirm)

        payload = deepcopy(base)
        payload.update(override)
        payload.update(
            {
                'experiment_name': f'soil_state_{name}_full_s111',
                'seed': 111,
                'epochs': 20,
                'max_updates_per_epoch': 5000,
                'save_weights_every': 5,
            }
        )
        write_config(args.output_dir / f'{name}_full_s111.yml', payload)

    smoke = deepcopy(base)
    smoke.update(experiments['s1'])
    smoke.update(
        {
            'experiment_name': 'soil_state_s1_smoke',
            'seed': 111,
            'epochs': 1,
            'max_updates_per_epoch': 5,
            'save_weights_every': 1,
            'batch_size': 8,
            'experimental_train_sample_stride': 24,
        }
    )
    write_config(args.output_dir / 's1_smoke.yml', smoke)
    print(f'wrote 16 configs to {args.output_dir}')


if __name__ == '__main__':
    main()
