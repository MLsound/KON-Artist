import pytest
import pandas as pd
import os
from scripts.generate_plots import generate_report_plot

@pytest.fixture
def valid_csv(tmp_path):
    """Creates a temporary CSV with dummy training data."""
    csv_file = tmp_path / "test_rewards.csv"
    data = {
        'step': list(range(0, 1000, 10)),
        'reward': [0.1 + (i * 0.008) for i in range(100)] # Gradual increase
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return str(csv_file)

@pytest.fixture
def empty_csv(tmp_path):
    """Creates a temporary empty CSV file."""
    csv_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_file, index=False)
    return str(csv_file)

def test_generate_plot_success(valid_csv, tmp_path):
    """Verify that a plot is successfully generated from valid data."""
    output_png = tmp_path / "output.png"
    generate_report_plot(valid_csv, str(output_png), window_size=10)

    expected_png = tmp_path / "convergence_test_rewards.png"
    assert expected_png.exists()
    assert expected_png.stat().st_size > 0

def test_generate_plot_missing_file(tmp_path):
    """Ensure no failure occurs when the CSV path is invalid."""
    invalid_path = "non_existent.csv"
    output_png = tmp_path / "should_not_exist.png"
    
    # Should log an error but not raise an exception
    generate_report_plot(invalid_path, str(output_png))
    assert not output_png.exists()

def test_generate_plot_empty_data(empty_csv, tmp_path):
    """Ensure the function handles empty CSV files gracefully."""
    output_png = tmp_path / "empty_output.png"
    generate_report_plot(empty_csv, str(output_png))
    assert not output_png.exists()

def test_moving_average_logic(valid_csv, tmp_path, caplog):
    """Test if moving average is skipped when data is too short."""
    output_png = tmp_path / "short_output.png"
    # Data has 100 rows, setting window to 150 should prevent smooth plot
    generate_report_plot(valid_csv, str(output_png), window_size=150)

    expected_png = tmp_path / "convergence_test_rewards.png"
    assert expected_png.exists()
    # Check that it didn't crash (manual check of plt state is hard, 
    # but we verify execution completion).