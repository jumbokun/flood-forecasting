#!/usr/bin/env python3
"""Build an immutable, validated DWD composite archive for 2001--2024.

Existing verified payloads can be reused with hard links. Missing files are
downloaded atomically. The manifest is only published after every requested
year/depth passes size, checksum, NetCDF shape, date, CRS, unit, and licence
checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from netCDF4 import Dataset, num2date


BASE_URL = (
    'https://opendata.dwd.de/climate_environment/CDC/grids_germany/'
    'daily/soil_moisture/composite'
)
DEFAULT_YEARS = tuple(range(2001, 2025))
DEFAULT_DEPTHS = ('0-10', '0-30')


def filename_for(year: int, depth: str) -> str:
    return f'grids_germany_daily_soil_moisture_composite_{year}_{depth}_v1.nc'


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def remote_metadata(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.head(url, timeout=(30, 120), allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get('Content-Length')
    if not length:
        raise RuntimeError(f'remote object has no Content-Length: {url}')
    return {
        'content_length': int(length),
        'etag': response.headers.get('ETag'),
        'last_modified': response.headers.get('Last-Modified'),
    }


def download_atomic(
    session: requests.Session,
    url: str,
    destination: Path,
    expected_size: int,
) -> None:
    partial = destination.with_suffix(destination.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
    if offset and response.status_code != 206:
        response.close()
        partial.unlink(missing_ok=True)
        offset = 0
        response = session.get(url, stream=True, timeout=(30, 300))
    response.raise_for_status()
    with partial.open('ab' if offset else 'wb') as target:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f'download size mismatch for {url}: {actual_size} != {expected_size}'
        )
    partial.replace(destination)


def reuse_or_download(
    session: requests.Session,
    source: Path,
    destination: Path,
    url: str,
    expected_size: int,
) -> str:
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError(
                f'wrong-sized existing destination: {destination}'
            )
        return 'reused_destination'
    if source.exists() and source.stat().st_size == expected_size:
        try:
            os.link(source, destination)
            return 'hardlinked_existing_archive'
        except OSError:
            shutil.copy2(source, destination)
            return 'copied_existing_archive'
    download_atomic(session, url, destination, expected_size)
    return 'downloaded'


def validate_netcdf(
    path: Path, expected_year: int, expected_depth: str
) -> dict[str, Any]:
    with Dataset(path) as dataset:
        for name in ('paws', 'time', 'x', 'y', 'transverse_mercator'):
            if name not in dataset.variables:
                raise RuntimeError(f'{path}: missing variable {name}')
        paws = dataset.variables['paws']
        time = dataset.variables['time']
        dates = num2date(
            time[:],
            units=time.units,
            calendar=getattr(time, 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
        )
        date_values = [item.date() for item in dates]
        expected_days = []
        cursor = date(expected_year, 1, 1)
        stop = date(expected_year + 1, 1, 1)
        while cursor < stop:
            expected_days.append(cursor)
            cursor += timedelta(days=1)
        if sorted(date_values) != expected_days:
            missing = sorted(set(expected_days).difference(date_values))
            duplicates = sorted(
                value
                for value in set(date_values)
                if date_values.count(value) > 1
            )
            raise RuntimeError(
                f'{path}: time coordinate is not a unique complete calendar; '
                f'missing={missing[:10]}, duplicates={duplicates[:10]}'
            )
        disorder_positions = [
            index
            for index in range(len(date_values) - 1)
            if date_values[index] >= date_values[index + 1]
        ]
        if paws.shape[-3] != len(dates) or paws.shape[-2:] != (866, 654):
            raise RuntimeError(f'{path}: unexpected paws shape {paws.shape}')
        title = str(getattr(dataset, 'title', ''))
        if expected_depth.replace('-', ' to ') not in title:
            raise RuntimeError(f'{path}: title/depth mismatch: {title!r}')
        if getattr(paws, 'units', None) != '% nFK':
            raise RuntimeError(f'{path}: unexpected paws units')
        crs = dataset.variables['transverse_mercator']
        if int(getattr(crs, 'epsg', -1)) != 31467:
            raise RuntimeError(f'{path}: expected EPSG:31467')
        licence = str(getattr(dataset, 'licence', ''))
        if 'CC-BY 4.0' not in licence:
            raise RuntimeError(f'{path}: unexpected licence {licence!r}')
        return {
            'paws_shape': list(paws.shape),
            'paws_dimensions': list(paws.dimensions),
            'paws_units': paws.units,
            'first_valid_time': min(dates).isoformat(),
            'last_valid_time': max(dates).isoformat(),
            'time_coordinate_monotonic': not disorder_positions,
            'time_disorder_positions': disorder_positions,
            'crs': 'EPSG:31467',
            'licence': licence,
            'creation_date': getattr(dataset, 'creation_date', None),
            'source': getattr(dataset, 'source', None),
            'model_version': getattr(dataset, 'model_version', None),
        }


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = (int(item) for item in part.split('-', 1))
            years.update(range(start, end + 1))
        else:
            years.add(int(part))
    return sorted(years)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--reuse-root', type=Path)
    parser.add_argument('--years', default='2001-2024')
    parser.add_argument('--depths', default=','.join(DEFAULT_DEPTHS))
    parser.add_argument('--base-url', default=BASE_URL)
    parser.add_argument('--manifest', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    depths = [item.strip() for item in args.depths.split(',') if item.strip()]
    if not years or any(year < 1900 or year > 2100 for year in years):
        raise ValueError('invalid years')
    if not depths or any(depth not in DEFAULT_DEPTHS for depth in depths):
        raise ValueError(f'depths must be a subset of {DEFAULT_DEPTHS}')

    output_root = args.output_root.resolve()
    reuse_root = (
        args.reuse_root.resolve() if args.reuse_root else Path('/__missing__')
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or output_root / 'manifest.json'
    if manifest.resolve().parent != output_root:
        raise ValueError('manifest must be directly inside output-root')

    session = requests.Session()
    session.headers.update(
        {'User-Agent': 'FloodWaive-Germany-soil-state-research/1.0'}
    )
    records: list[dict[str, Any]] = []
    for year in years:
        for depth in depths:
            filename = filename_for(year, depth)
            url = f'{args.base_url.rstrip("/")}/{year}/{filename}'
            remote = remote_metadata(session, url)
            destination = output_root / filename
            action = reuse_or_download(
                session=session,
                source=reuse_root / filename,
                destination=destination,
                url=url,
                expected_size=remote['content_length'],
            )
            netcdf = validate_netcdf(destination, year, depth)
            record = {
                'year': year,
                'depth_cm': depth,
                'filename': filename,
                'url': url,
                'bytes': destination.stat().st_size,
                'sha256': sha256_file(destination),
                'materialization': action,
                'remote': remote,
                'netcdf': netcdf,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        'year': year,
                        'depth_cm': depth,
                        'materialization': action,
                        'bytes': record['bytes'],
                    }
                ),
                flush=True,
            )

    payload = {
        'schema': 'dwd-composite-full-archive-v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_base_url': args.base_url,
        'licence': 'CC BY 4.0',
        'years': years,
        'depths_cm': depths,
        'file_count': len(records),
        'logical_payload_bytes': sum(item['bytes'] for item in records),
        'reuse_root': str(reuse_root) if args.reuse_root else None,
        'files': records,
    }
    partial_manifest = manifest.with_suffix(manifest.suffix + '.part')
    partial_manifest.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    partial_manifest.replace(manifest)
    print(
        json.dumps(
            {
                'manifest': str(manifest),
                **{
                    k: payload[k]
                    for k in ('file_count', 'logical_payload_bytes')
                },
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
