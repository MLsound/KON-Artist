import pytest
import torch
from unittest.mock import MagicMock, patch
from src.models.aasist import AASISTWrapper

@pytest.fixture
def mock_waveform():
    """Returns a dummy waveform tensor [Batch, Samples]."""
    return torch.randn(1, 16000)

@pytest.fixture
def wrapper():
    """
    Initializes the wrapper with a mocked model to avoid 
    loading a real file during testing.
    """
    with patch("torch.load"), patch("src.utils.logger.get_logger"):
        # We mock _load_model to return a MagicMock instead of a bare Module
        AASISTWrapper._load_model = MagicMock(return_value=MagicMock())
        return AASISTWrapper(model_path="dummy.pth", device="cpu")

def test_score_calculation_logic(wrapper, mock_waveform):
    """Verify that the softmax score is correctly extracted from logits."""
    # Mock model output: Logits for [Spoof, Bonafide] and a dummy embedding
    # Logits [0, 0] should result in a 0.5 probability (50/50)
    mock_logits = torch.tensor([[0.0, 0.0]])
    mock_embedding = torch.randn(1, 160)
    
    # We use a side effect to simulate the hook firing during the forward pass.
    # Since get_score_and_embedding resets self._embedding to None at the start,
    # it must be set DURING the model call.
    def mock_forward(x):
        wrapper._embedding = mock_embedding
        return mock_logits
        
    wrapper.model.side_effect = mock_forward
    
    scores, embedding = wrapper.get_score_and_embedding(mock_waveform)
    
    assert scores[0] == pytest.approx(0.5)
    assert torch.equal(embedding, mock_embedding)

def test_device_fallback(mock_waveform):
    """Ensure the wrapper handles device assignment correctly."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.load"), \
         patch.object(AASISTWrapper, '_load_model'):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # We need to mock _setup_hooks to avoid it failing if it can't find out_layer
        with patch.object(AASISTWrapper, '_setup_hooks'):
            inst = AASISTWrapper(model_path="dummy.pth", device=device)
            assert inst.device == "cpu"

def test_get_score_no_grad(wrapper, mock_waveform):
    """Ensure inference is performed without gradient tracking."""
    mock_logits = torch.tensor([[1.0, 2.0]])
    mock_embedding = torch.randn(1, 160)

    def mock_forward(x):
        wrapper._embedding = mock_embedding
        return mock_logits

    wrapper.model.side_effect = mock_forward

    # If no_grad is working, the output shouldn't have a grad_fn
    scores, embedding = wrapper.get_score_and_embedding(mock_waveform)
    assert embedding.grad_fn is None

def test_return_logits_option(wrapper, mock_waveform):
    """Verify that get_score_and_embedding correctly returns logits when return_logits=True."""
    import math
    mock_logits = torch.tensor([[-5.0, 3.5]])
    mock_embedding = torch.randn(1, 160)
    
    def mock_forward(x):
        wrapper._embedding = mock_embedding
        return mock_logits
        
    wrapper.model.side_effect = mock_forward
    
    scores, logits, embedding = wrapper.get_score_and_embedding(mock_waveform, return_logits=True)
    
    assert logits[0] == pytest.approx(3.5)
    assert scores[0] == pytest.approx(math.exp(3.5) / (math.exp(-5.0) + math.exp(3.5)))
    assert torch.equal(embedding, mock_embedding)