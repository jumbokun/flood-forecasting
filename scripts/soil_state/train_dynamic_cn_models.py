#!/usr/bin/env python3
"""Train interpretable dynamic-CN models without opening the double holdout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn
from torch.nn import functional as functional


MODELS = ('B0', 'B1', 'M1', 'M2', 'N0')
TARGETS = {
    'lh': 'direct_runoff_lh_mm',
    'eckhardt': 'direct_runoff_eckhardt_mm',
}
SEEDS = (111, 222, 333)
LAMBDA_VALUES = (0.05, 0.20)
SHUFFLE_SEED = 20260805
STATIC_FEATURES = (
    'sand_0_30cm_mean',
    'silt_0_30cm_mean',
    'clay_0_30cm_mean',
    'coarse_fragments_0_30cm_mean',
    'bulk_density_0_30cm_mean',
    'soil_organic_carbon_0_30cm_mean',
    'artificial_surfaces_perc',
    'agricultural_areas_perc',
    'forests_and_seminatural_areas_perc',
    'wetlands_perc',
    'water_bodies_perc',
    'area',
    'elev_mean',
    'elev_min',
    'elev_max',
)
B0_FEATURES = (
    'sand_0_30cm_mean',
    'silt_0_30cm_mean',
    'clay_0_30cm_mean',
    'coarse_fragments_0_30cm_mean',
    'bulk_density_0_30cm_mean',
    'soil_organic_carbon_0_30cm_mean',
    'artificial_surfaces_perc',
    'agricultural_areas_perc',
    'forests_and_seminatural_areas_perc',
    'wetlands_perc',
    'water_bodies_perc',
    'log_area',
    'elev_mean',
    'relief',
    'month_sin',
    'month_cos',
)
B1_EXTRA_FEATURES = ('log_antecedent_rain_5d', 'log_antecedent_rain_30d')


def parse_csv_values(value: str, converter: Any = str) -> list[Any]:
    return [converter(item.strip()) for item in value.split(',') if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--targets', default=','.join(TARGETS))
    parser.add_argument(
        '--lambdas', default=','.join(str(value) for value in LAMBDA_VALUES)
    )
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in SEEDS))
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--torch-threads', type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def verify_open_input(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / 'artifact_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = {
        item['path']: item for item in manifest['files']
    }
    record = records.get('events_open.parquet')
    if record is None:
        raise RuntimeError('dataset manifest does not bind events_open.parquet')
    open_path = dataset_root / 'events_open.parquet'
    actual = sha256(open_path)
    if actual != record['sha256'] or open_path.stat().st_size != record['bytes']:
        raise RuntimeError('events_open.parquet failed manifest verification')
    return {
        'path': str(open_path.resolve()),
        'bytes': open_path.stat().st_size,
        'sha256': actual,
    }


def susceptibility_mask(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen, outcome-blind M2 susceptibility rules."""
    water_repellent = (
        (frame['sand_0_30cm_mean'] >= 31.67)
        & (frame['soil_organic_carbon_0_30cm_mean'] >= 46.37)
        & (frame['forests_and_seminatural_areas_perc'] >= 38.52)
    )
    crusting = (
        (frame['agricultural_areas_perc'] >= 65.14)
        & (frame['silt_0_30cm_mean'] >= 47.35)
    )
    high_clay = frame['clay_0_30cm_mean'] >= 27.02
    wet_or_water = (
        frame['wetlands_perc'] + frame['water_bodies_perc'] >= 5.0
    )
    return pd.DataFrame(
        {
            'susceptible_water_repellent': water_repellent,
            'susceptible_crusting': crusting,
            'shrink_swell_clay_separate': high_clay,
            'wet_or_water_excluded': wet_or_water,
            'm2_susceptible': (water_repellent | crusting)
            & ~high_clay
            & ~wet_or_water,
        },
        index=frame.index,
    )


def grouped_shuffle(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    group_columns: tuple[str, ...] = ('basin', 'season', 'split'),
    seed: int = SHUFFLE_SEED,
) -> np.ndarray:
    """Shuffle rows only within frozen negative-control groups."""
    output = frame[list(columns)].to_numpy(dtype=np.float32, copy=True)
    grouped = frame.groupby(list(group_columns), sort=True, observed=True).indices
    for key, indices in grouped.items():
        positions = np.asarray(indices, dtype=np.int64)
        rng = np.random.default_rng(stable_seed(f'{seed}:{key}'))
        output[positions] = output[rng.permutation(positions)]
    return output


def prepare_model_frame(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only the open event table and construct leakage-safe features."""
    events = pd.read_parquet(dataset_root / 'events_open.parquet')
    if 'double_holdout' in set(events['split']):
        raise RuntimeError('open event table contains double-holdout rows')
    common = events[
        events['analysis_eligible_both']
        & (events['antecedent_rain_5d_coverage'] >= 0.95)
        & (events['antecedent_rain_30d_coverage'] >= 0.95)
    ].copy()
    attributes = pd.read_parquet(dataset_root / 'basin_attributes.parquet')
    attribute_columns = ['basin', 'federal_state', *STATIC_FEATURES]
    common = common.merge(
        attributes[attribute_columns],
        on='basin',
        how='left',
        validate='many_to_one',
    )
    if common[list(STATIC_FEATURES)].isna().all(axis=None):
        raise RuntimeError('all static features are missing after merge')
    month = common['event_start'].dt.month.to_numpy(dtype=np.float64)
    common['month_sin'] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    common['month_cos'] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    common['log_area'] = np.log1p(common['area'].clip(lower=0.0))
    common['relief'] = common['elev_max'] - common['elev_min']
    common['log_antecedent_rain_5d'] = np.log1p(
        common['antecedent_rain_5d_mm'].clip(lower=0.0)
    )
    common['log_antecedent_rain_30d'] = np.log1p(
        common['antecedent_rain_30d_mm'].clip(lower=0.0)
    )
    mask = susceptibility_mask(common)
    for column in mask:
        common[column] = mask[column]
    train = common[common['split'] == 'train']
    if train.empty:
        raise RuntimeError('training split is empty')
    state_season_p20 = train.groupby(
        ['federal_state', 'season'], observed=True
    )['dwd_soil_0_30_p50'].quantile(0.20)
    season_p20 = train.groupby('season', observed=True)[
        'dwd_soil_0_30_p50'
    ].quantile(0.20)
    thresholds = common['dwd_soil_0_30_expanding_p20'].astype(np.float64).copy()
    missing = thresholds.isna() | (common['dwd_history_days'] < 730)
    for row_index in common.index[missing]:
        key = (
            common.at[row_index, 'federal_state'],
            common.at[row_index, 'season'],
        )
        fallback = state_season_p20.get(key, math.nan)
        if not math.isfinite(fallback):
            fallback = season_p20[common.at[row_index, 'season']]
        thresholds.at[row_index] = fallback
    common['dry_threshold_model'] = thresholds
    common['dry_model'] = (
        common['dwd_soil_0_30_p50'] < common['dry_threshold_model']
    )
    common['dry_hinge'] = (
        (
            common['dry_threshold_model'] - common['dwd_soil_0_30_p50']
        ).clip(lower=0.0)
        / 100.0
        * common['m2_susceptible'].astype(float)
    )
    common['m2_effective_dry'] = (
        common['dry_model'] & common['m2_susceptible']
    )
    common = common.sort_values(['basin', 'event_start']).reset_index(drop=True)
    basin_q90 = common.groupby('basin')['direct_runoff_lh_mm'].transform(
        lambda values: values.quantile(0.90)
    )
    effective_dry = common['m2_effective_dry']
    audit = {
        'common_event_count': len(common),
        'common_basin_count': int(common['basin'].nunique()),
        'split_counts': {
            str(key): int(value)
            for key, value in common['split'].value_counts().items()
        },
        'split_basin_counts': {
            str(key): int(value)
            for key, value in common.groupby('split')['basin'].nunique().items()
        },
        'dry_event_count': int(common['dry_model'].sum()),
        'dry_basin_count': int(common.loc[common['dry_model'], 'basin'].nunique()),
        'm2_effective_dry_event_count': int(effective_dry.sum()),
        'm2_effective_dry_basin_count': int(
            common.loc[effective_dry, 'basin'].nunique()
        ),
        'm2_effective_dry_top_decile_lh_event_count': int(
            (effective_dry & (common['direct_runoff_lh_mm'] >= basin_q90)).sum()
        ),
        'susceptible_basin_count': int(
            common.loc[common['m2_susceptible'], 'basin'].nunique()
        ),
        'susceptible_development_basin_count': int(
            common.loc[
                common['m2_susceptible']
                & common['split'].isin(['train', 'validation', 'temporal_test']),
                'basin',
            ].nunique()
        ),
        'fallback_threshold_event_count': int(missing.sum()),
        'double_holdout_rows_read': 0,
    }
    return common, audit


def standardized_features(
    frame: pd.DataFrame, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, Any]]:
    train = frame['split'] == 'train'
    values = frame[list(feature_names)].to_numpy(dtype=np.float64)
    train_values = values[train]
    median = np.nanmedian(train_values, axis=0)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(median, np.where(missing)[1])
    mean = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    output = ((values - mean) / scale).astype(np.float32)
    return output, {
        'feature_names': list(feature_names),
        'imputation_median': median.tolist(),
        'mean': mean.tolist(),
        'scale': scale.tolist(),
        'imputed_value_count': int(missing.sum()),
    }


def scs_runoff_torch(
    precipitation: torch.Tensor,
    curve_number: torch.Tensor,
    abstraction_ratio: float,
) -> torch.Tensor:
    storage = 25400.0 / curve_number - 254.0
    excess = precipitation - abstraction_ratio * storage
    runoff = excess.square() / (
        precipitation + (1.0 - abstraction_ratio) * storage
    )
    return torch.where(excess > 0.0, runoff, torch.zeros_like(runoff))


class DynamicCN(nn.Module):
    """Low-capacity bounded-CN model with optional monotone state terms."""

    def __init__(self, base_feature_count: int, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.linear = nn.Linear(base_feature_count, 1)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
        if model_name in {'M1', 'M2', 'N0'}:
            self.wet_raw = nn.Parameter(
                torch.full((2,), -2.25) + 0.01 * torch.randn(2)
            )
        else:
            self.register_parameter('wet_raw', None)
        if model_name == 'M2':
            self.dry_raw = nn.Parameter(
                torch.tensor(-2.25) + 0.01 * torch.randn(())
            )
        else:
            self.register_parameter('dry_raw', None)

    def forward(
        self,
        base: torch.Tensor,
        wetness: torch.Tensor,
        dry_hinge: torch.Tensor,
    ) -> torch.Tensor:
        predictor = self.linear(base).squeeze(-1)
        if self.wet_raw is not None:
            predictor = predictor + (
                functional.softplus(self.wet_raw) * wetness
            ).sum(dim=1)
        if self.dry_raw is not None:
            predictor = predictor + functional.softplus(
                self.dry_raw
            ) * dry_hinge
        return 30.0 + 68.0 * torch.sigmoid(predictor)


def basin_weights(basins: pd.Series) -> np.ndarray:
    counts = basins.value_counts()
    weights = basins.map(lambda value: 1.0 / counts[value]).to_numpy(float)
    return (weights / weights.sum()).astype(np.float32)


def basin_balanced_mae(
    basins: np.ndarray, observed: np.ndarray, predicted: np.ndarray
) -> float:
    frame = pd.DataFrame(
        {
            'basin': basins,
            'absolute_error': np.abs(observed - predicted),
        }
    )
    return float(frame.groupby('basin')['absolute_error'].mean().median())


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def fit_one_model(
    frame: pd.DataFrame,
    base: np.ndarray,
    wetness: np.ndarray,
    dry_hinge: np.ndarray,
    target: np.ndarray,
    model_name: str,
    abstraction_ratio: float,
    seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    train_mask = (frame['split'] == 'train').to_numpy()
    validation_mask = (frame['split'] == 'validation').to_numpy()
    model = DynamicCN(base.shape[1], model_name)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    base_tensor = _tensor(base)
    wet_tensor = _tensor(wetness)
    dry_tensor = _tensor(dry_hinge)
    rain_tensor = _tensor(frame['event_rain_mm'].to_numpy())
    target_tensor = _tensor(target)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask))
    validation_indices = np.flatnonzero(validation_mask)
    train_weight = _tensor(
        basin_weights(frame.loc[train_mask, 'basin']).reshape(-1)
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation = math.inf
    epochs_without_improvement = 0
    final_loss = math.nan
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        cn = model(
            base_tensor[train_indices],
            wet_tensor[train_indices],
            dry_tensor[train_indices],
        )
        predicted = scs_runoff_torch(
            rain_tensor[train_indices], cn, abstraction_ratio
        )
        element_loss = functional.smooth_l1_loss(
            predicted,
            target_tensor[train_indices],
            beta=1.0,
            reduction='none',
        )
        loss = (element_loss * train_weight).sum()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        model.eval()
        with torch.no_grad():
            validation_cn = model(
                base_tensor[validation_indices],
                wet_tensor[validation_indices],
                dry_tensor[validation_indices],
            )
            validation_prediction = scs_runoff_torch(
                rain_tensor[validation_indices],
                validation_cn,
                abstraction_ratio,
            ).numpy()
        validation_mae = basin_balanced_mae(
            frame.loc[validation_mask, 'basin'].to_numpy(),
            target[validation_mask],
            validation_prediction,
        )
        if validation_mae < best_validation - 1e-7:
            best_validation = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError(f'{model_name} failed to produce a valid state')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_cn = model(base_tensor, wet_tensor, dry_tensor)
        all_prediction = scs_runoff_torch(
            rain_tensor, all_cn, abstraction_ratio
        )
    state = {
        key: value.detach().cpu().numpy().tolist()
        for key, value in model.state_dict().items()
    }
    if model.wet_raw is not None:
        state['wet_nonnegative_coefficients'] = functional.softplus(
            model.wet_raw
        ).detach().numpy().tolist()
    if model.dry_raw is not None:
        state['dry_nonnegative_coefficient'] = float(
            functional.softplus(model.dry_raw).detach()
        )
    training = {
        'seed': seed,
        'best_epoch_zero_based': best_epoch,
        'epochs_run': epoch + 1,
        'best_validation_basin_balanced_mae_mm': best_validation,
        'final_training_loss': final_loss,
    }
    return all_prediction.numpy(), all_cn.numpy(), state, training


def metric_record(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = frame.loc[mask, ['basin']].copy()
    selected['observed'] = observed[mask]
    selected['predicted'] = predicted[mask]
    selected['absolute_error'] = np.abs(
        selected['observed'] - selected['predicted']
    )
    basin_mae = selected.groupby('basin')['absolute_error'].mean()
    return {
        'events': len(selected),
        'basins': int(selected['basin'].nunique()),
        'event_mae_mm': float(selected['absolute_error'].mean()),
        'basin_balanced_mae_mm': float(basin_mae.median()),
        'mean_observed_runoff_mm': float(selected['observed'].mean()),
        'mean_predicted_runoff_mm': float(selected['predicted'].mean()),
    }


def paired_comparison(
    frame: pd.DataFrame,
    observed: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    bootstrap_replicates: int,
    seed_key: str,
) -> dict[str, Any]:
    selected = frame.loc[mask, ['basin']].copy()
    selected['observed'] = observed[mask]
    selected['reference_error'] = np.abs(
        observed[mask] - reference[mask]
    )
    selected['candidate_error'] = np.abs(
        observed[mask] - candidate[mask]
    )
    by_basin = selected.groupby('basin').agg(
        reference_mae=('reference_error', 'mean'),
        candidate_mae=('candidate_error', 'mean'),
    )
    reference_mae = float(by_basin['reference_mae'].median())
    candidate_mae = float(by_basin['candidate_mae'].median())
    improvement = (
        (reference_mae - candidate_mae) / reference_mae
        if reference_mae > 0.0
        else math.nan
    )
    rng = np.random.default_rng(stable_seed(seed_key))
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    values = by_basin[['reference_mae', 'candidate_mae']].to_numpy()
    for index in range(bootstrap_replicates):
        sampled = values[rng.integers(0, len(values), len(values))]
        ref = np.median(sampled[:, 0])
        cand = np.median(sampled[:, 1])
        bootstrap[index] = (ref - cand) / ref if ref > 0.0 else np.nan
    try:
        p_value = float(
            wilcoxon(
                by_basin['reference_mae'],
                by_basin['candidate_mae'],
                alternative='greater',
                zero_method='zsplit',
            ).pvalue
        )
    except ValueError:
        p_value = 1.0
    selected['top_decile'] = selected['observed'] >= selected.groupby(
        'basin'
    )['observed'].transform(lambda values_: values_.quantile(0.90))
    top = selected[selected['top_decile']]
    reference_under = float(
        (reference[mask][selected['top_decile'].to_numpy()] <= 0.5 * top['observed']).mean()
    )
    candidate_under = float(
        (candidate[mask][selected['top_decile'].to_numpy()] <= 0.5 * top['observed']).mean()
    )
    return {
        'events': len(selected),
        'basins': len(by_basin),
        'reference_basin_balanced_mae_mm': reference_mae,
        'candidate_basin_balanced_mae_mm': candidate_mae,
        'relative_mae_improvement': improvement,
        'bootstrap_ci_low': float(np.nanquantile(bootstrap, 0.025)),
        'bootstrap_ci_high': float(np.nanquantile(bootstrap, 0.975)),
        'fraction_basins_better': float(
            (by_basin['candidate_mae'] < by_basin['reference_mae']).mean()
        ),
        'one_sided_wilcoxon_p': p_value,
        'top_decile_events': len(top),
        'reference_severe_underprediction_rate': reference_under,
        'candidate_severe_underprediction_rate': candidate_under,
    }


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    p_values = values.to_numpy(dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(len(values), dtype=float)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def evaluate_m2_sample_gate(data_audit: dict[str, Any]) -> dict[str, Any]:
    gate = {
        'minimum_events': 500,
        'minimum_basins': 100,
        'minimum_top_decile_events': 100,
        'observed_events': data_audit['m2_effective_dry_event_count'],
        'observed_basins': data_audit['m2_effective_dry_basin_count'],
        'observed_top_decile_events': data_audit[
            'm2_effective_dry_top_decile_lh_event_count'
        ],
    }
    gate['passed'] = bool(
        gate['observed_events'] >= gate['minimum_events']
        and gate['observed_basins'] >= gate['minimum_basins']
        and gate['observed_top_decile_events']
        >= gate['minimum_top_decile_events']
    )
    return gate


def main() -> None:
    args = parse_args()
    models = parse_csv_values(args.models)
    requested_models = list(models)
    targets = parse_csv_values(args.targets)
    lambdas = parse_csv_values(args.lambdas, float)
    seeds = parse_csv_values(args.seeds, int)
    if not set(models).issubset(MODELS) or not set(targets).issubset(TARGETS):
        raise ValueError('unknown model or target')
    if not seeds or not lambdas:
        raise ValueError('seeds and lambdas must not be empty')
    if args.torch_threads < 1 or args.torch_threads > 8:
        raise ValueError('torch-threads must be in [1, 8]')
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError('output-root already exists')
    output_root.mkdir(parents=True)
    models_root = output_root / 'models'
    models_root.mkdir()
    torch.set_num_threads(args.torch_threads)
    source_contract = verify_open_input(args.dataset_root)
    frame, data_audit = prepare_model_frame(args.dataset_root)
    m2_sample_gate = evaluate_m2_sample_gate(data_audit)
    m2_stopped_before_fit = 'M2' in models and not m2_sample_gate['passed']
    if m2_stopped_before_fit:
        models.remove('M2')
        print(
            json.dumps(
                {
                    'model': 'M2',
                    'status': 'STOPPED_BEFORE_FIT_SAMPLE_GATE',
                    'sample_gate': m2_sample_gate,
                }
            ),
            flush=True,
        )
    base_b0, scaler_b0 = standardized_features(frame, B0_FEATURES)
    base_b1, scaler_b1 = standardized_features(
        frame, (*B0_FEATURES, *B1_EXTRA_FEATURES)
    )
    wetness = frame[
        ['dwd_soil_0_10_p50', 'dwd_soil_0_30_p50']
    ].to_numpy(dtype=np.float32) / 100.0
    shuffled_wetness = grouped_shuffle(
        frame,
        ('dwd_soil_0_10_p50', 'dwd_soil_0_30_p50'),
    ) / 100.0
    dry_hinge = frame['dry_hinge'].to_numpy(dtype=np.float32)
    zero_hinge = np.zeros(len(frame), dtype=np.float32)
    prediction_records: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    ensemble_predictions: dict[tuple[str, float, str], np.ndarray] = {}
    for target_name in targets:
        target_column = TARGETS[target_name]
        observed = frame[target_column].to_numpy(dtype=np.float32)
        for abstraction_ratio in lambdas:
            for model_name in models:
                base = base_b0 if model_name == 'B0' else base_b1
                scaler = scaler_b0 if model_name == 'B0' else scaler_b1
                model_wetness = (
                    shuffled_wetness if model_name == 'N0' else wetness
                )
                model_hinge = dry_hinge if model_name == 'M2' else zero_hinge
                seed_predictions: list[np.ndarray] = []
                seed_curve_numbers: list[np.ndarray] = []
                configuration = (
                    f'{target_name}_lambda{int(round(abstraction_ratio * 100)):03d}'
                )
                model_dir = models_root / configuration / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                for seed in seeds:
                    predicted, curve_number, state, training = fit_one_model(
                        frame,
                        base,
                        model_wetness,
                        model_hinge,
                        observed,
                        model_name,
                        abstraction_ratio,
                        seed,
                        args.max_epochs,
                        args.patience,
                        args.learning_rate,
                        args.weight_decay,
                    )
                    seed_predictions.append(predicted)
                    seed_curve_numbers.append(curve_number)
                    write_json(
                        model_dir / f'seed_{seed}.json',
                        {
                            'model': model_name,
                            'target': target_name,
                            'target_column': target_column,
                            'initial_abstraction_ratio': abstraction_ratio,
                            'training': training,
                            'state': state,
                            'scaler': scaler,
                        },
                    )
                ensemble = np.mean(seed_predictions, axis=0)
                ensemble_cn = np.mean(seed_curve_numbers, axis=0)
                ensemble_predictions[
                    (target_name, abstraction_ratio, model_name)
                ] = ensemble
                for split in ('validation', 'temporal_test', 'spatial_test'):
                    split_mask = (frame['split'] == split).to_numpy()
                    for subset, subset_mask in (
                        ('all', split_mask),
                        ('dry', split_mask & frame['dry_model'].to_numpy()),
                        (
                            'm2_effective_dry',
                            split_mask
                            & frame['m2_effective_dry'].to_numpy(),
                        ),
                        ('not_dry', split_mask & ~frame['dry_model'].to_numpy()),
                    ):
                        if not subset_mask.any():
                            continue
                        metric_records.append(
                            {
                                'target': target_name,
                                'initial_abstraction_ratio': abstraction_ratio,
                                'model': model_name,
                                'split': split,
                                'subset': subset,
                                **metric_record(
                                    frame, observed, ensemble, subset_mask
                                ),
                            }
                        )
                prediction_records.append(
                    pd.DataFrame(
                        {
                            'event_id': frame['event_id'],
                            'basin': frame['basin'],
                            'event_start': frame['event_start'],
                            'split': frame['split'],
                            'dry_model': frame['dry_model'],
                            'm2_effective_dry': frame['m2_effective_dry'],
                            'm2_susceptible': frame['m2_susceptible'],
                            'target': target_name,
                            'initial_abstraction_ratio': abstraction_ratio,
                            'model': model_name,
                            'observed_direct_runoff_mm': observed,
                            'predicted_direct_runoff_mm': ensemble,
                            'predicted_curve_number': ensemble_cn,
                        }
                    )
                )
                print(
                    json.dumps(
                        {
                            'target': target_name,
                            'lambda': abstraction_ratio,
                            'model': model_name,
                            'seeds_complete': len(seeds),
                        }
                    ),
                    flush=True,
                )
    metrics = pd.DataFrame(metric_records)
    metrics_path = output_root / 'open_split_metrics.parquet'
    metrics.to_parquet(metrics_path, index=False, compression='zstd')
    predictions = pd.concat(prediction_records, ignore_index=True)
    predictions_path = output_root / 'open_predictions.parquet'
    predictions.to_parquet(predictions_path, index=False, compression='zstd')
    comparisons: list[dict[str, Any]] = []
    five_year: list[dict[str, Any]] = []
    for target_name in targets:
        observed = frame[TARGETS[target_name]].to_numpy(dtype=np.float32)
        for abstraction_ratio in lambdas:
            validation = metrics[
                (metrics['target'] == target_name)
                & (metrics['initial_abstraction_ratio'] == abstraction_ratio)
                & (metrics['split'] == 'validation')
                & (metrics['subset'] == 'all')
                & metrics['model'].isin(['B0', 'B1'])
            ]
            baseline = str(
                validation.sort_values('basin_balanced_mae_mm').iloc[0]['model']
            )
            reference = ensemble_predictions[
                (target_name, abstraction_ratio, baseline)
            ]
            for split in ('temporal_test', 'spatial_test'):
                split_mask = (frame['split'] == split).to_numpy()
                for candidate_name in ('M1', 'N0'):
                    if candidate_name not in models:
                        continue
                    comparisons.append(
                        {
                            'target': target_name,
                            'initial_abstraction_ratio': abstraction_ratio,
                            'split': split,
                            'comparison': f'{candidate_name}_vs_{baseline}',
                            'subset': 'all',
                            'reference_model': baseline,
                            'candidate_model': candidate_name,
                            **paired_comparison(
                                frame,
                                observed,
                                reference,
                                ensemble_predictions[
                                    (
                                        target_name,
                                        abstraction_ratio,
                                        candidate_name,
                                    )
                                ],
                                split_mask,
                                args.bootstrap_replicates,
                                f'{target_name}:{abstraction_ratio}:{split}:{candidate_name}',
                            ),
                        }
                    )
                if 'M1' in models and 'M2' in models:
                    for subset, subset_mask in (
                        (
                            'dry',
                            split_mask
                            & frame['m2_effective_dry'].to_numpy(),
                        ),
                        (
                            'not_dry',
                            split_mask
                            & frame['m2_susceptible'].to_numpy()
                            & ~frame['dry_model'].to_numpy(),
                        ),
                    ):
                        comparisons.append(
                            {
                                'target': target_name,
                                'initial_abstraction_ratio': abstraction_ratio,
                                'split': split,
                                'comparison': 'M2_vs_M1',
                                'subset': subset,
                                'reference_model': 'M1',
                                'candidate_model': 'M2',
                                **paired_comparison(
                                    frame,
                                    observed,
                                    ensemble_predictions[
                                        (target_name, abstraction_ratio, 'M1')
                                    ],
                                    ensemble_predictions[
                                        (target_name, abstraction_ratio, 'M2')
                                    ],
                                    subset_mask,
                                    args.bootstrap_replicates,
                                    f'{target_name}:{abstraction_ratio}:{split}:M2:{subset}',
                                ),
                            }
                        )
            if 'M1' in models and 'M2' in models:
                spatial = frame['split'] == 'spatial_test'
                for start, end in (
                    (2001, 2005),
                    (2006, 2010),
                    (2011, 2015),
                    (2016, 2020),
                    (2021, 2023),
                ):
                    block = (
                        spatial
                        & frame['m2_effective_dry']
                        & frame['event_start'].dt.year.between(start, end)
                    ).to_numpy()
                    five_year.append(
                        {
                            'target': target_name,
                            'initial_abstraction_ratio': abstraction_ratio,
                            'block': f'{start}-{end}',
                            **paired_comparison(
                                frame,
                                observed,
                                ensemble_predictions[
                                    (target_name, abstraction_ratio, 'M1')
                                ],
                                ensemble_predictions[
                                    (target_name, abstraction_ratio, 'M2')
                                ],
                                block,
                                args.bootstrap_replicates,
                                f'{target_name}:{abstraction_ratio}:block:{start}',
                            ),
                        }
                    )
    comparison_frame = pd.DataFrame(comparisons)
    m2_family = (
        (comparison_frame['comparison'] == 'M2_vs_M1')
        & (comparison_frame['subset'] == 'dry')
    )
    comparison_frame['bh_q_value'] = np.nan
    comparison_frame.loc[m2_family, 'bh_q_value'] = benjamini_hochberg(
        comparison_frame.loc[m2_family, 'one_sided_wilcoxon_p']
    )
    comparison_path = output_root / 'open_model_comparisons.parquet'
    comparison_frame.to_parquet(
        comparison_path, index=False, compression='zstd'
    )
    five_year_path = output_root / 'm2_spatial_five_year_blocks.parquet'
    pd.DataFrame(
        five_year,
        columns=[
            'target',
            'initial_abstraction_ratio',
            'block',
            'events',
            'basins',
            'reference_basin_balanced_mae_mm',
            'candidate_basin_balanced_mae_mm',
            'relative_mae_improvement',
            'bootstrap_ci_low',
            'bootstrap_ci_high',
            'fraction_basins_better',
            'one_sided_wilcoxon_p',
            'top_decile_events',
            'reference_severe_underprediction_rate',
            'candidate_severe_underprediction_rate',
        ],
    ).to_parquet(
        five_year_path, index=False, compression='zstd'
    )
    complete_grid = (
        tuple(requested_models) == MODELS
        and (
            tuple(models) == MODELS
            or (m2_stopped_before_fit and tuple(models) == ('B0', 'B1', 'M1', 'N0'))
        )
        and tuple(targets) == tuple(TARGETS)
        and tuple(lambdas) == LAMBDA_VALUES
        and tuple(seeds) == SEEDS
        and args.max_epochs == 500
        and args.patience == 50
    )
    summary = {
        'schema': 'germany-dynamic-cn-open-model-run-v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'complete_preregistered_grid': complete_grid,
        'double_holdout_opened': False,
        'source_open_event_table': source_contract,
        'data_audit': data_audit,
        'm2_sample_gate': m2_sample_gate,
        'm2_stopped_before_fit': m2_stopped_before_fit,
        'requested_models': requested_models,
        'models': models,
        'targets': targets,
        'initial_abstraction_ratios': lambdas,
        'seeds': seeds,
        'optimizer': {
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'basin_balanced_smooth_l1_beta_mm': 1.0,
        },
        'prediction_rows': len(predictions),
        'comparison_rows': len(comparison_frame),
        'note': (
            'No file below dataset_root/double_holdout was read. Open-test '
            'results may select a frozen candidate, but cannot satisfy the '
            'double-holdout promotion gate.'
        ),
    }
    summary_path = output_root / 'training_summary.json'
    write_json(summary_path, summary)
    contract_path = output_root / 'run_contract.json'
    write_json(
        contract_path,
        {
            'dataset_root': str(args.dataset_root.resolve()),
            'output_root': str(output_root),
            'builder': str(Path(__file__).resolve()),
            'models': models,
            'targets': targets,
            'lambdas': lambdas,
            'seeds': seeds,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'bootstrap_replicates': args.bootstrap_replicates,
            'torch_threads': args.torch_threads,
        },
    )
    manifest_files = [
        metrics_path,
        predictions_path,
        comparison_path,
        five_year_path,
        summary_path,
        contract_path,
    ]
    model_files = sorted(models_root.rglob('seed_*.json'))
    manifest_files.extend(model_files)
    manifest = {
        'schema': 'germany-dynamic-cn-open-model-run-manifest-v1',
        'files': [
            {
                'path': str(path.relative_to(output_root)),
                'bytes': path.stat().st_size,
                'sha256': sha256(path),
            }
            for path in manifest_files
        ],
        'model_json_count': len(model_files),
        'double_holdout_opened': False,
    }
    write_json(output_root / 'artifact_manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
