import pytest
import pandas as pd
import numpy as np
from src.diagnostics.dashboard_manifold import interpolate_surface

def test_interpolate_surface_aggregation_collision_x():
    # Create mock cluster data where x_col == z_col to test index collision
    data = {
        "dsp_tilt": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2],
        "dsp_harmonics": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0],
        "cluster": [1, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Run interpolation with x_col == z_col
    gx, gy, gz = interpolate_surface(
        df_cluster=df,
        x_col="dsp_tilt",
        y_col="dsp_harmonics",
        z_col="dsp_tilt",
        grid_size=10
    )
    
    assert gx is not None
    assert gy is not None
    assert gz is not None
    assert gx.shape == (10, 10)
    assert gy.shape == (10, 10)
    assert gz.shape == (10, 10)

def test_interpolate_surface_aggregation_collision_y():
    # Create mock cluster data where y_col == z_col to test index collision
    data = {
        "dsp_tilt": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2],
        "dsp_harmonics": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0],
        "cluster": [1, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Run interpolation with y_col == z_col
    gx, gy, gz = interpolate_surface(
        df_cluster=df,
        x_col="dsp_tilt",
        y_col="dsp_harmonics",
        z_col="dsp_harmonics",
        grid_size=10
    )
    
    assert gx is not None
    assert gy is not None
    assert gz is not None
    assert gx.shape == (10, 10)
    assert gy.shape == (10, 10)
    assert gz.shape == (10, 10)

def test_interpolate_surface_no_collision():
    # Create mock cluster data with distinct x, y, and z columns
    data = {
        "dsp_tilt": [0.1, 0.2, 0.3, 0.4, 0.5],
        "dsp_harmonics": [1.0, 2.0, 3.0, 4.0, 5.0],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "cluster": [1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Run interpolation
    gx, gy, gz = interpolate_surface(
        df_cluster=df,
        x_col="dsp_tilt",
        y_col="dsp_harmonics",
        z_col="score",
        grid_size=10
    )
    
    assert gx is not None
    assert gy is not None
    assert gz is not None
    assert gx.shape == (10, 10)
    assert gy.shape == (10, 10)
    assert gz.shape == (10, 10)

def test_column_normalization_logic():
    # Create mock dataframe with abnormal capitalization and whitespace
    data = {
        "  Cluster  ": [1, 2, 3],
        "Score": [0.9, 0.8, 0.7],
        "DSP_Tilt": [0.1, 0.2, 0.3]
    }
    df = pd.DataFrame(data)
    
    # Simulate the defensive validation hook logic
    if "cluster" not in df.columns:
        df.columns = [col.lower().strip() for col in df.columns]
        
    assert "cluster" in df.columns
    assert "score" in df.columns
    assert "dsp_tilt" in df.columns
    assert list(df.columns) == ["cluster", "score", "dsp_tilt"]
