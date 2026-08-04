from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / 'scripts'
    / 'soil_state'
    / 'build_dwd_full_archive.py'
)
SPEC = importlib.util.spec_from_file_location('build_dwd_full_archive', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_years_range_and_list():
    assert MODULE.parse_years('2001-2003,2005,2003') == [2001, 2002, 2003, 2005]


def test_filename_contract():
    assert MODULE.filename_for(2024, '0-30') == (
        'grids_germany_daily_soil_moisture_composite_2024_0-30_v1.nc'
    )


def test_validate_existing_official_file():
    path = Path(
        '/data/zhu/forecast-discharge-alarm/02_data_forcing/dwd-soil-moisture/'
        'historical_event_years_v1/'
        'grids_germany_daily_soil_moisture_composite_2024_0-10_v1.nc'
    )
    if not path.exists():
        return
    result = MODULE.validate_netcdf(path, 2024, '0-10')
    assert result['paws_units'] == '% nFK'
    assert result['crs'] == 'EPSG:31467'
    assert result['first_valid_time'].startswith('2024-01-01')
    assert result['last_valid_time'].startswith('2024-12-31')
    assert result['time_coordinate_monotonic'] is False
    assert result['time_disorder_positions'] == [233]
