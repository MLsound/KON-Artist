import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from scripts.baseline_evaluate import evaluate
from src.models.aasist import AASISTWrapper

@pytest.fixture
def mock_protocol_content():
    """Returns a dummy ASVspoof protocol string."""
    return "LA_0001 LA_E_10001 - - bonafide\nLA_0001 LA_E_10002 - - spoof\n"

@pytest.fixture
def wrapper():
    """Initializes a wrapper with a mocked model architecture."""
    with patch("torch.load"), patch("src.utils.logger.get_logger"):
        AASISTWrapper._load_model = MagicMock(return_value=MagicMock())
        return AASISTWrapper(model_path="dummy.pth", device="cpu")

def test_wrapper_score_extraction(wrapper):
    """Verify that scores are correctly processed from model logits."""
    # Mock model output: Logits [0, 0] result in a 0.5 probability
    mock_logits = torch.tensor([[0.0, 0.0]])
    mock_embedding = torch.randn(1, 160)
    
    def mock_forward(x):
        wrapper._embedding = mock_embedding
        return mock_logits
        
    wrapper.model.side_effect = mock_forward
    
    # get_score_and_embedding returns (scores, embedding) where scores is a numpy array
    scores, embedding = wrapper.get_score_and_embedding(torch.randn(1, 16000))
    assert scores[0] == pytest.approx(0.5)
    assert embedding.shape == (1, 160)

@patch("scripts.baseline_evaluate.AASISTWrapper")
@patch("scripts.baseline_evaluate.torchaudio.load")
@patch("scripts.baseline_evaluate.compute_eer")
@patch("os.path.exists", return_value=True)
def test_full_evaluation_workflow(
    mock_exists, mock_eer, mock_audio, mock_wrapper_class, 
    mock_protocol_content, tmp_path
):
    """Tests the entire evaluate() function logic using mocks."""
    # 1. Setup mocks
    mock_detector = mock_wrapper_class.return_value
    mock_detector.get_score_and_embedding.side_effect = [([0.9], None), ([0.1], None)]
    mock_audio.return_value = (torch.randn(1, 16000), 16000)
    mock_eer.return_value = (0.05, np.array([0.1]), np.array([0.1]), np.array([0.5])) # EER, fpr, fnr, thresholds
    
    protocol_file = tmp_path / "test_protocol.txt"
    protocol_file.write_text(mock_protocol_content)
    
    # 2. Run evaluation
    eer, _ = evaluate(
        model_path="dummy.pth",
        data_dir=str(tmp_path),
        protocol_path=str(protocol_file),
        scores_out=str(tmp_path / "scores.txt"),
        device="cpu"
    )
    
    # 3. Assertions
    assert eer == 0.05
    assert mock_detector.get_score_and_embedding.call_count == 2
    assert (tmp_path / "scores.txt").exists()