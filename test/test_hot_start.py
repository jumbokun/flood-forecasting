import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import numpy as np
from googlehydrology.utils.config import Config
from googlehydrology.datasetzoo.multimet import Multimet
from googlehydrology.modelzoo.handoff_forecast_lstm import HandoffForecastLSTM
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM

def get_base_cfg(tmp_path: Path):
    data_dir = "/usr/local/google/home/gsnearing/wabash_subset_test"
    basin_file = Path(data_dir) / "basins.txt"
    # Make sure we only use one basin for the test
    with open(tmp_path / "test_basin.txt", "w") as f:
        f.write("us_03338780\n")

    cfg_dict = {
        'model': 'HandoffForecastLSTM',
        'run_dir': str(tmp_path),
        'experiment_name': 'test_hot_start',
        'data_dir': data_dir,
        'test_basin_file': str(tmp_path / "test_basin.txt"),
        'train_basin_file': str(tmp_path / "test_basin.txt"),
        'validation_basin_file': str(tmp_path / "test_basin.txt"),
        'test_start_date': '01/01/2012',
        'test_end_date': '05/01/2012', # Few dates
        'train_start_date': '01/01/2011',
        'train_end_date': '31/12/2011',
        'validation_start_date': '01/01/2011',
        'validation_end_date': '31/12/2011',
        'seq_length': 30,
        'lead_time': 5,
        'head': 'regression',
        'target_variables': ['streamflow'],
        'hindcast_inputs': ['pr_day_gridmet', 'tmmn_day_gridmet'],
        'forecast_inputs': ['pr_day_gridmet', 'tmmn_day_gridmet'],
        'static_attributes': ['area_gages2', 'elev_mean_x', 'p_mean_x'],
        'hidden_size': 16,
        'forecast_overlap': 2,
        'state_handoff_network': {'type': 'fc', 'hiddens': [16], 'activation': ['relu'], 'dropout': 0.0},
        'hindcast_embedding': {'type': 'fc', 'hiddens': [16], 'activation': ['relu'], 'dropout': 0.0},
        'forecast_embedding': {'type': 'fc', 'hiddens': [16], 'activation': ['relu'], 'dropout': 0.0},
        'statics_embedding': {'type': 'fc', 'hiddens': [16], 'activation': ['relu'], 'dropout': 0.0},
        'lazy_load': False,
        'batch_size': 2,
        'epochs': 1,
        'initial_learning_rate': 1e-3,
        'loss': 'nse',
        'optimizer': 'Adam',
        'number_of_basins': 1,
        'nan_handling_method': 'masked_mean',
    }
    return cfg_dict

def test_handoff_forecast_lstm_hot_start(tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'HandoffForecastLSTM'
    cfg = Config(cfg_dict, dev_mode=True)
    
    # We load dataset to get real data (Multimet)
    dataset = Multimet(cfg, is_train=False, period='test', compute_scaler=False)
    
    model = HandoffForecastLSTM(cfg)
    model.eval()

    # Get one sample
    sample = dataset[0]
    # Add batch dimension
    data = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else v for k, v in sample.items()}
    data['x_d_hindcast'] = {k: torch.tensor(v).unsqueeze(0) for k, v in sample['x_d_hindcast'].items()}
    data['x_d_forecast'] = {k: torch.tensor(v).unsqueeze(0) for k, v in sample['x_d_forecast'].items()}

    state_path = tmp_path / "state.npz"
    with torch.no_grad():
        model.save_state(data, state_path)
        cold_preds = model(data)
        
    # Hot start run
    cfg.hot_start_path = str(state_path)
    with torch.no_grad():
        hot_preds = model(data)
        
    diff = (cold_preds['y_hat'] - hot_preds['y_hat'][:, cfg.seq_length:, :]).abs().max()
    # Note: handoff output is size [batch, seq_length+lead_time, targets], wait...
    # handoff cold output is [batch, seq_length+lead_time, targets? No, `(:, -self.seq_length :, :)` in `forward`]
    # Let's just compare the last lead_time predictions
    cold_last = cold_preds['y_hat'][:, -cfg.lead_time:, :]
    hot_last = hot_preds['y_hat'][:, -cfg.lead_time:, :]
    assert (cold_last - hot_last).abs().max() < 1e-5

def test_mean_embedding_forecast_lstm_hot_start(tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'MeanEmbeddingForecastLSTM'
    cfg_dict['n_distributions'] = 1
    cfg = Config(cfg_dict, dev_mode=True)
    
    # Use real basin data
    dataset = Multimet(cfg, is_train=False, period='test', compute_scaler=False)
    
    model = MeanEmbeddingForecastLSTM(cfg)
    model.eval()

    sample = dataset[0]
    data = {k: torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else v for k, v in sample.items()}
    data['x_d_hindcast'] = {k: torch.tensor(v).unsqueeze(0) for k, v in sample['x_d_hindcast'].items()}
    data['x_d_forecast'] = {k: torch.tensor(v).unsqueeze(0) for k, v in sample['x_d_forecast'].items()}

    state_path = tmp_path / "state_mean.npz"
    with torch.no_grad():
        model.save_state(data, state_path)
        cold_preds = model(data)
        
    cfg.hot_start_path = str(state_path)
    with torch.no_grad():
        hot_preds = model(data)
        
    # mean embedding returns full seq_length + lead_time
    cold_last = cold_preds['y_hat'][:, -cfg.lead_time:, :]
    hot_last = hot_preds['y_hat'][:, -cfg.lead_time:, :]
    assert (cold_last - hot_last).abs().max() < 1e-5

