
import pytest
import numpy as np
import torch
import os
import pickle
from unittest.mock import MagicMock, patch
from src.data.clustering import extract_all_embeddings, run_clustering_pipeline
from src.env.audio_attack import AudioAttackEnv
from src.env.wrappers import VecDetectorWrapper

class MockDetector:
    def get_score_and_embedding(self, waveform):
        # Return a random score and a 160-dim embedding
        batch_size = waveform.shape[0] if len(waveform.shape) > 1 else 1
        return torch.rand(batch_size), torch.randn(batch_size, 160)

@pytest.fixture
def mock_clustering_pipeline():
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=2)),
        ('gmm', GaussianMixture(n_components=4, random_state=42))
    ])
    # Fit on some dummy data
    dummy_data = np.random.randn(10, 160)
    pipeline.fit(dummy_data)
    return pipeline

def test_extract_all_embeddings():
    detector = MockDetector()
    # Mock loader to return 5 samples
    mock_loader = [ (torch.randn(1, 64600), 1) for _ in range(5) ]
    
    with patch('src.data.clustering.generator_from_ds', return_value=mock_loader):
        embeddings = extract_all_embeddings(detector, mock_loader)
        
    assert embeddings.shape == (5, 160)
    assert embeddings.dtype == np.float32

def test_audio_attack_env_clustering(mock_clustering_pipeline, tmp_path):
    # Setup mock clustering config
    model_path = tmp_path / "gmm_registry.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(mock_clustering_pipeline, f)
        
    clustering_config = {
        "n_components": 4,
        "model_path": str(model_path)
    }
    
    detector = MockDetector()
    env = AudioAttackEnv(detector=detector, clustering_config=clustering_config)
    
    # Verify observation space
    assert env.observation_space.shape == (164,)
    
    # Mock data stream to avoid actual loading
    env.audio_files = MagicMock()
    # Mock generator yielding (waveform, label)
    env.audio_files.__next__.return_value = (torch.randn(1, 64600), "Spoof")
    
    obs, info = env.reset()
    assert obs.shape == (164,)
    assert obs.dtype == np.float32
    
    # Check step
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (164,)
    assert obs.dtype == np.float32

def test_vec_detector_wrapper_clustering(mock_clustering_pipeline, tmp_path):
    model_path = tmp_path / "gmm_registry.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(mock_clustering_pipeline, f)
        
    config = {
        "clustering": {
            "n_components": 4,
            "model_path": str(model_path)
        }
    }
    
    # Mock Venv
    mock_venv = MagicMock()
    mock_venv.num_envs = 2
    mock_venv.reset.return_value = np.random.randn(2, 64600)
    mock_venv.step_wait.return_value = (np.random.randn(2, 64600), np.zeros(2), np.zeros(2), [{}, {}])
    
    detector = MockDetector()
    wrapper = VecDetectorWrapper(mock_venv, detector=detector, config=config)
    
    assert wrapper.observation_space.shape == (164,)
    
    obs = wrapper.reset()
    assert obs.shape == (2, 164)
    assert obs.dtype == np.float32
    
    obs, rewards, dones, infos = wrapper.step(np.random.randn(2, 7))
    assert obs.shape == (2, 164)
    assert obs.dtype == np.float32
    assert "terminal_observation" in infos[0] or dones[0] == False # terminal_obs only added if done
