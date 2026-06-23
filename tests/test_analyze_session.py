import os
import pandas as pd
import pytest
from unittest.mock import patch
from scripts.analyze_session import main, extract_id_from_filename, get_winners

def test_extract_id_from_filename():
    assert extract_id_from_filename("rewards_20260603_203248.csv") == "_20260603_203248"
    assert extract_id_from_filename("no_date.csv") == ""

def test_get_winners():
    df = pd.DataFrame({
        "score": [0.4, 0.5, 0.6],
        "dsp_tilt": [0.1, 0.2, 0.3]
    })
    winners = get_winners(df, 0.5)
    assert len(winners) == 2
    assert list(winners["score"]) == [0.5, 0.6]

def test_analyze_session_main(tmp_path):
    csv_file = tmp_path / "rewards_20260603_203248.csv"
    data = {
        "score": [0.3, 0.6, 0.8],
        "bonus": [1, 2, 3],
        "dsp_tilt": [-0.5, 0.0, 0.5],
        "dsp_harmonics": [0.1, 0.5, 0.9]
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    out_dir = tmp_path / "analysis"
    
    test_args = [
        "scripts/analyze_session.py",
        "-f", str(csv_file),
        "-t", "0.5",
        "--out-dir", str(out_dir)
    ]
    
    with patch("sys.argv", test_args):
        main()
        
    winners_path = out_dir / "winners_20260603_203248.csv"
    analysis_path = out_dir / "analysis_20260603_203248.csv"
    
    assert winners_path.exists()
    assert analysis_path.exists()
    
    analysis_df = pd.read_csv(analysis_path)
    assert "dsp_parameter" in analysis_df.columns
    assert list(analysis_df["dsp_parameter"]) == ["dsp_harmonics", "dsp_tilt"]
    
    tilt_row = analysis_df[analysis_df["dsp_parameter"] == "dsp_tilt"].iloc[0]
    assert tilt_row["mean"] == 0.25
    assert tilt_row["min"] == 0.0
    assert tilt_row["max"] == 0.5
