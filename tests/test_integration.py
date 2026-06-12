import pytest
import os
import numpy as np
import torch
import torchaudio
from unittest.mock import MagicMock, patch
from src.env.audio_attack import AudioAttackEnv
from src.models.aasist import AASISTWrapper
from src.utils.audio import preprocess_audio

@pytest.fixture
def integration_setup(tmp_path):
    """
    Sets up the full RL environment for integration testing.
    The internal model loader is mocked to bypass HF Hub and heavy weights.
    """
    with patch("src.models.aasist.AASISTWrapper._load_model") as mock_load_model, \
         patch("src.utils.logger.get_logger"):
        
        # 1. Setup the Mocked Detector Core
        mock_model_instance = MagicMock()
        mock_load_model.return_value = mock_model_instance
        
        # 2. Instantiate the Wrapper (uses the mock)
        wrapper = AASISTWrapper(model_path="dummy.pth", device="cpu")

        # 3. Use side_effect to simulate hook firing
        def mock_forward(x):
            wrapper._embedding = torch.randn(x.shape[0], 160)
            return torch.tensor([[0.9, 0.1]])
        mock_model_instance.side_effect = mock_forward

        # 4. Load a real audio sample from the dataset
        sample_path = './samples_spoof_01.wav'
        # Verify the sample exists before proceeding
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Could not find sample at {sample_path}")
        
        # 4. Pass the path as a list of tuples to the environment
        # We simulate the generator behavior here
        waveform, sr = torchaudio.load(sample_path)
        processed = preprocess_audio(waveform, sr)
        
        env = AudioAttackEnv(
            detector=wrapper,
            dsp_config={},
            audio_files=[(processed, 0)]
        )
        
        return env, wrapper


def test_full_env_step_cycle(integration_setup):
    """
    Verifies the end-to-end flow:
    Action -> Denormalize -> DSP -> AASIST3 -> Reward
    """
    env, wrapper = integration_setup
    
    # 1. Reset Phase
    obs, info = env.reset()
    assert obs.shape == (160,)
    assert isinstance(obs, np.ndarray)
    
    # 2. Step Phase
    # RL Action in range [-1, 1]
    action = np.array([0.5, -0.2, 0.0, 0.1, -0.5, 0.8, -1.0], dtype=np.float32)
    
    # We mock apply_codec_simulation to avoid backend dependencies (FFmpeg)
    with patch.object(env.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        obs, reward, terminated, truncated, info = env.step(action)
    
    # 3. Functional Assertions
    # Reward should be a float
    assert isinstance(reward, float)
    assert obs.shape == (160,)
    assert terminated is False
    
    # Verify that the internal components were called
    assert wrapper.model.called
    assert env.last_dsp_params is not None
    assert 'bitrate' in env.last_dsp_params

def test_dsp_modification_persistence(integration_setup):
    """
    Ensures the DSP pipeline actually modifies the audio waveform 
    before it reaches the detector.
    """
    env, _ = integration_setup
    env.reset()
    
    # Action designed to high-tilt and high-jitter the audio
    action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    
    with patch.object(env.detector, 'get_score_and_embedding') as mock_inference, \
         patch.object(env.dsp, 'apply_codec_simulation', side_effect=lambda x, **kwargs: x):
        
        # Capture the waveform passed to the detector
        mock_inference.return_value = ([0.5], torch.randn(1, 160))
        env.step(action)
        
        processed_waveform = mock_inference.call_args[0][0]
        
        # Verification: The processed audio should differ from initial random noise
        # (Assuming the DSP pipeline isn't a simple Identity function)
        assert not torch.equal(processed_waveform, env.current_audio)
