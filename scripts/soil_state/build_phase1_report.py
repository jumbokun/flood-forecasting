#!/usr/bin/env python3
"""Build the auditable Phase-1 Germany soil-state experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-dir', required=True, type=Path)
    parser.add_argument('--confirmation-dir', required=True, type=Path)
    parser.add_argument('--daily-audit', required=True, type=Path)
    parser.add_argument('--sampling-audit', required=True, type=Path)
    parser.add_argument('--hourly-audit', required=True, type=Path)
    parser.add_argument('--screen-manifest', required=True, type=Path)
    parser.add_argument('--confirmation-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def format_number(value: float, digits: int = 3) -> str:
    return f'{value:.{digits}f}'


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join('---' for _ in columns) + ' |',
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


def paired_digits(metric: str) -> int:
    return 6 if metric in {'rmse_mm_h', 'mae_mm_h'} else 3


def format_paired_interval(row: pd.Series) -> str:
    digits = paired_digits(str(row['metric']))
    low = row['bootstrap_median_ci_low']
    high = row['bootstrap_median_ci_high']
    return f'[{low:.{digits}f}, {high:.{digits}f}]'


def main() -> None:
    args = parse_args()
    screen = read_json(args.screen_dir / 'screen_conclusion.json')[
        'screen_gate'
    ]
    confirmation_payload = read_json(
        args.confirmation_dir / 'confirmation_conclusion.json'
    )
    confirmation = confirmation_payload['confirmation_gate']
    daily = read_json(args.daily_audit)
    sampling = read_json(args.sampling_audit)
    hourly = read_json(args.hourly_audit)

    seed_summary = pd.read_csv(
        args.confirmation_dir / 'confirmation_seed_summary.csv'
    )
    band = seed_summary[seed_summary['band'] == 'forecast_25_48h'].copy()
    band_table = pd.DataFrame(
        {
            'Variant': band['variant'].str.upper(),
            'Median NSE': band['median_nse'].map(format_number),
            'Median KGE': band['median_kge'].map(format_number),
            'RMSE (mm/h)': band['median_rmse_mm_h'].map(format_number),
            'Absolute FHV (%)': band['median_abs_fhv_percent'].map(
                lambda value: format_number(value, 1)
            ),
        }
    )

    paired = pd.read_csv(
        args.confirmation_dir / 'confirmation_paired_bootstrap.csv'
    )
    paired = paired[
        (paired['band'] == 'forecast_25_48h')
        & paired['metric'].isin(['nse', 'kge', 'rmse_mm_h', 'abs_fhv_percent'])
    ].copy()
    paired_table = pd.DataFrame(
        {
            'Comparison': paired['comparison'].str.replace('_', ' '),
            'Metric': paired['metric'],
            'Paired median gain': paired.apply(
                lambda row: format_number(
                    row['paired_median_delta'], paired_digits(row['metric'])
                ),
                axis=1,
            ),
            '95% bootstrap interval': paired.apply(
                format_paired_interval,
                axis=1,
            ),
            'Basins better': paired['fraction_basins_s1_better'].map(
                lambda value: f'{100 * value:.1f}%'
            ),
        }
    )

    screen_c0 = screen['comparisons']['s1_minus_c0']
    screen_c1 = screen['comparisons']['s1_minus_c1']
    go = confirmation['status'] == 'GO_TO_PHASE2'
    decision = (
        'GO to Phase 2 full multi-seed training and ablation'
        if go
        else 'NO-GO / inconclusive: do not spend the Phase 2 budget yet'
    )
    production = 'This is not a production release and not a learned Curve Number mapping.'

    report = f"""# Germany antecedent soil-state encoder — Phase 0/1

Date: 2026-08-04

Decision: **{decision}.**

Boundary: **{production}**

## What we tested

The scientific question was narrowed from “convert DWD soil moisture to CN” to a
falsifiable precursor: does an issue-time-safe DWD antecedent soil-state signal add
out-of-sample discharge forecast skill beyond meteorological history alone?

Three matched national CAMELS-DE-1h models were compared:

| Code | Inputs | Purpose |
| --- | --- | --- |
| C0 | 720 h RADKLIM history + ICON forecast | Base model |
| C1 | 1,440 h RADKLIM history + ICON forecast | Controls for merely giving the model more history |
| S1 | C0 + ten causal DWD soil-state features | Tests DWD information gain |

All confirmation runs used seed 111, five epochs and 2,000 updates per epoch. Training
used 1,189 basins through 2022, validation used 2023, and the held-out 297-basin test
covered 2024. Evaluation uses one 00 UTC issue time per day and separate lead hours
0–48; valid time is asserted to equal issue time plus lead.

## Phase 0 data contract

- DWD archive: 2001–2024, both 0–10 cm and 0–30 cm layers, 48 official NetCDF files.
- Basin product: {daily['basin_count']:,} basins, {daily['date_count']:,} daily states,
  P10/P50/P90 and coverage for both layers plus a layer gradient.
- Causal hourly replay: {hourly['hourly_start']} to {hourly['hourly_end']},
  {hourly['availability_lag_hours']} h conservative publication lag; feature age is
  {hourly['state_age_hours_min']:.0f}–{hourly['state_age_hours_max']:.0f} h.
- Spatial approximation audit: {sampling['comparison_count']:,} comparisons; MAE
  {sampling['mae_percent_nfk']:.2f} %nFK, P95 absolute error
  {sampling['p95_absolute_error_percent_nfk']:.2f} %nFK. The predeclared P95 ≤10
  %nFK gate passed.

DWD `%nFK` is a model-derived plant-available-water state, not an infiltration
measurement. The 48 h replay lag deliberately prevents a future-valid daily raster
from leaking into an earlier issue time.

## Screening result

The deliberately cheap 3-seed screen passed its promotion rule. In the 25–48 h band,
S1 beat C0 in {screen_c0['positive_seed_count']}/3 matched seeds and C1 in
{screen_c1['positive_seed_count']}/3. Median seed-level paired NSE gains were
{screen_c0['median_of_seed_deltas']:+.4f} against C0 and
{screen_c1['median_of_seed_deltas']:+.4f} against C1.

## Confirmation result (held-out 2024)

Positive paired gain always means S1 is better. The decision gate requires the
25–48 h NSE median gain and its basin-bootstrap 95% interval to be above zero against
both C0 and C1.

{markdown_table(band_table)}

{markdown_table(paired_table)}

Gate output: **{confirmation['status']}**.

## Interpretation

{confirmation['rule']}

The confirmation result fails that rule. S1 is worse than C0 across every lead band,
and its 25–48 h NSE degradation against C0 excludes zero. This does not show that
antecedent moisture is irrelevant; it shows that the present raw daily DWD feature
bundle, lag contract and short training recipe do not justify a larger model spend.
Possible causes include stale daily states, redundancy with the 30-day rainfall history,
model-derived rather than observed moisture, and optimization difficulty from the extra
inputs.

It also does not support a universal infiltration law, the proposed U-shaped
dry/medium/wet response, or a direct soil-moisture-to-CN conversion.

## Actions before reconsidering Phase 2

1. Use the existing predictions for event stratification by antecedent dryness,
   rainfall intensity, soil class, land use, season and catchment scale. This is the
   cheapest way to test whether a useful subgroup signal is hidden by national medians.
2. Confirm actual DWD publication timestamps. The 48 h replay lag is conservative but
   synthetic and may discard useful recency.
3. Run only small targeted ablations: 0–10 vs 0–30 cm, P50-only vs quantiles,
   standardized anomaly vs raw `%nFK`, and 24/48/72 h lag assumptions.
4. Reopen full multi-seed training only if one of those diagnostics gives a stable,
   predeclared signal. Only after robust event-level gains should a constrained state
   encoder or effective runoff/CN diagnostic head be attempted.

## Reproducibility and limitations

- Exact checkpoints, configs, source hashes, runtime hashes and audit hashes are frozen
  in `{args.screen_manifest}` and `{args.confirmation_manifest}`.
- The current hourly model runtime includes uncommitted platform work; its exact file
  hashes are frozen rather than silently copied into this branch.
- DWD 2023/2024 official 0–10 cm files contain non-monotonic time ordering; the builder
  sorts dates and verifies the unique calendar.
- Up to 256 grid-cell centers represent each basin. P95 error passed, but the observed
  maximum audit error was {sampling['max_absolute_error_percent_nfk']:.2f} %nFK.
- Confirmation has one seed and is intentionally budget-limited. It failed the gate,
  so full training is not authorized by this result.

## Primary references and open-source context

- [DWD daily 1 km soil-moisture composite specification](https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/soil_moisture/composite/BESCHREIBUNG_grids_germany_daily_soil_moisture_composite_de.pdf)
- [DWD soil-moisture viewer](https://www.dwd.de/DE/leistungen/bofeu_viewer/bofeuviewer.html)
- [DWD explanation of `%nFK` and AMBAV](https://www.dwd.de/DE/service/lexikon/Functions/glossar.html?lv2=100310&lv3=787548)
- [mHM distributed hydrological model](https://www.ufz.de/index.php?en=40114)
- [LARSIM operational water-balance model](https://larsim.info/)
- Doerr et al. (2000), *Soil water repellency: causes, characteristics and
  hydro-geomorphological significance*, Earth-Science Reviews 51, 33–65,
  https://doi.org/10.1016/S0012-8252(00)00011-8
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
