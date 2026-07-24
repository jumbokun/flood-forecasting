import pytest
from unittest.mock import patch
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
        'predict_last_n': 0,
    }
    return cfg_dict

@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_handoff_forecast_lstm_hot_start(mock_check, mock_load, tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'HandoffForecastLSTM'
    cfg = Config(cfg_dict, dev_mode=True)
    model = HandoffForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {}
    
    data['x_d_hindcast'] = {
        'pr_day_gridmet': torch.rand(batch_size, cfg.seq_length + cfg.lead_time, 1, device=device),
        'tmmn_day_gridmet': torch.rand(batch_size, cfg.seq_length + cfg.lead_time, 1, device=device)
    }
    data['x_d_forecast'] = {
        'pr_day_gridmet': torch.rand(batch_size, cfg.lead_time + cfg.forecast_overlap, 1, device=device),
        'tmmn_day_gridmet': torch.rand(batch_size, cfg.lead_time + cfg.forecast_overlap, 1, device=device)
    }
    data['x_s'] = torch.rand(batch_size, len(cfg.static_attributes), device=device)

    state_path = tmp_path / "state_handoff.npz"
    with torch.no_grad():
        model.seq_length = cfg.seq_length - 1
        model.save_state(data, state_path)
        model.seq_length = cfg.seq_length
        cold_preds = model(data)
        
    # Hot start run with seq_length=1
    cfg._cfg['seq_length'] = 1
    cfg.hot_start_path = str(state_path)
    model.seq_length = 1
    
    hot_data = {}
    hot_data['x_d_hindcast'] = {
        k: v[:, -(cfg.lead_time + 1):, :] for k, v in data['x_d_hindcast'].items()
    }
    hot_data['x_d_forecast'] = data['x_d_forecast']
    hot_data['x_s'] = data['x_s']

    with torch.no_grad():
        hot_preds = model(hot_data)
        
    cold_last = cold_preds['y_hat'][:, -cfg.lead_time:, :]
    hot_last = hot_preds['y_hat'][:, -cfg.lead_time:, :]
    assert (cold_last - hot_last).abs().max() < 1e-5

@patch('googlehydrology.datautils.scaler.Scaler.load')
@patch('googlehydrology.datautils.scaler.Scaler.check_zero_scale')
def test_mean_embedding_forecast_lstm_hot_start(mock_check, mock_load, tmp_path):
    cfg_dict = get_base_cfg(tmp_path)
    cfg_dict['model'] = 'MeanEmbeddingForecastLSTM'
    cfg_dict['n_distributions'] = 1
    # For MeanEmbedding, forecast_overlap is typically sequence length, so it gets seq_length + lead_time points.
    cfg_dict['forecast_overlap'] = cfg_dict['seq_length']
    cfg = Config(cfg_dict, dev_mode=True)
    model = MeanEmbeddingForecastLSTM(cfg)
    model.eval()

    device = next(model.parameters()).device
    batch_size = 2
    data = {}
    
    data['x_d_hindcast'] = {
        'pr_day_gridmet': torch.rand(batch_size, cfg.seq_length + cfg.lead_time, 1, device=device),
        'tmmn_day_gridmet': torch.rand(batch_size, cfg.seq_length + cfg.lead_time, 1, device=device)
    }
    data['x_d_forecast'] = {
        'pr_day_gridmet': torch.rand(batch_size, cfg.seq_length + cfg.lead_time, 1, device=device),
        'tmmn_day_gridmet': torch.rand(batch_size, cfg.seq_length + cfg.lead_time, 1, device=device)
    }
    data['x_s'] = torch.rand(batch_size, len(cfg.static_attributes), device=device)

    state_path = tmp_path / "state_mean.npz"
    with torch.no_grad():
        model.seq_length = cfg.seq_length - 1
        model.save_state(data, state_path)
        model.seq_length = cfg.seq_length
        cold_preds = model(data)
        
    # Hot start run with seq_length=1
    cfg._cfg['seq_length'] = 1
    cfg.hot_start_path = str(state_path)
    model.seq_length = 1

    hot_data = {}
    hot_data['x_d_hindcast'] = {
        k: v[:, -(cfg.lead_time + 1):, :] for k, v in data['x_d_hindcast'].items()
    }
    hot_data['x_d_forecast'] = {
        k: v[:, -(cfg.lead_time + 1):, :] for k, v in data['x_d_forecast'].items()
    }
    hot_data['x_s'] = data['x_s']

    with torch.no_grad():
        hot_preds = model(hot_data)
        
    cold_last = cold_preds['y_hat'][:, -cfg.lead_time:, :]
    hot_last = hot_preds['y_hat'][:, -cfg.lead_time:, :]
    assert (cold_last - hot_last).abs().max() < 1e-5

