import os
import re
import pandas as pd
import pytest
from unittest.mock import patch, mock_open
from src.diagnostics.dashboard_manifold import (
    get_hyperparameters,
    find_convergence_image,
    load_data,
    interpolate_global_surface,
    scan_winners_files
)

def test_get_hyperparameters_mocked():
    mock_log_content = (
        "Some irrelevant logs\n"
        "Training configuration: N_STEPS=2048, EPOCHS=10, BATCH=512, TOTAL_UPDATES=50, TOTAL_TIMESTEPS=102400\n"
        "Another log line\n"
    )
    
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open(read_data=mock_log_content)):
        params = get_hyperparameters("20260614_144009")
        assert params is not None
        assert params["TOTAL_TIMESTEPS"] == "102400"
        assert params["N_STEPS"] == "2048"
        assert params["TOTAL_UPDATES"] == "50"
        assert params["EPOCHS"] == "10"
        assert params["BATCH"] == "512"

def test_find_convergence_image_mocked():
    def mock_exists(path):
        return "outputs/convergence_20260614_144009.jpg" in path
        
    with patch("os.path.exists", side_effect=mock_exists):
        img_path = find_convergence_image("20260614_144009")
        assert img_path is not None
        assert img_path.endswith("outputs/convergence_20260614_144009.jpg")

def test_load_data(tmp_path):
    csv_file = tmp_path / "winners_20260614_144009.csv"
    df_data = {
        "DSP_TILT": [0.1, 0.2],
        "DSP_HARMONICS": [0.5, 0.6],
        "Score": [0.85, 0.92],
        "Cluster": [1.0, 2.0]
    }
    pd.DataFrame(df_data).to_csv(csv_file, index=False)
    
    loaded_df = load_data(str(csv_file))
    
    assert list(loaded_df.columns) == ["dsp_tilt", "dsp_harmonics", "score", "cluster"]
    assert loaded_df["cluster"].dtype == int
    assert list(loaded_df["cluster"]) == [1, 2]

def test_interpolate_global_surface():
    df_data = {
        "dsp_tilt": [0.1, 0.1, 0.2, 0.2],
        "dsp_harmonics": [0.5, 0.6, 0.5, 0.6],
        "score": [0.8, 0.9, 0.75, 0.85],
        "cluster": [1, 1, 2, 2]
    }
    df = pd.DataFrame(df_data)
    
    gx, gy, gz, gs = interpolate_global_surface(df, "dsp_tilt", "dsp_harmonics", "score")
    assert gx is not None
    assert gy is not None
    assert gz is not None
    assert gs is not None

def test_scan_winners_files_mocked():
    mock_files = ["outputs/winners_20260614_144009.csv"]
    with patch("glob.glob", return_value=mock_files):
        files = scan_winners_files()
        assert files is not None
        assert len(files) == 1
        assert files[0].endswith("outputs/winners_20260614_144009.csv")
