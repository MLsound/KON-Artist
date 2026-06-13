import pytest
import torch
import numpy as np
import gymnasium as gym
from src.models.custom_policies import SquashedGaussianDistribution, SquashedGaussianActorCriticPolicy

def test_squashed_gaussian_distribution():
    action_dim = 3
    dist = SquashedGaussianDistribution(action_dim=action_dim)
    
    # Create mock parameters
    mean = torch.tensor([[0.0, 0.5, -0.5]])
    log_std = torch.tensor([[0.0, 0.0, 0.0]])
    
    dist.proba_distribution(mean, log_std)
    
    # 1. Test sample boundaries
    samples = dist.sample()
    assert samples.shape == (1, action_dim)
    assert torch.all(samples >= -1.0) and torch.all(samples <= 1.0)
    
    # 2. Test mode mapping (tanh(mean))
    expected_mode = torch.tanh(mean)
    assert torch.allclose(dist.mode(), expected_mode)
    
    # 3. Test log_prob calculation and Jacobian correction
    # Sample a mock action
    action = torch.tensor([[0.2, 0.4, -0.6]])
    log_prob = dist.log_prob(action)
    
    # Manual computation
    clipped_action = torch.clamp(action, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    raw_action = 0.5 * torch.log((1.0 + clipped_action) / (1.0 - clipped_action))
    
    # Base gaussian log prob
    gaussian_log_prob = dist.gaussian_dist.log_prob(raw_action).sum(dim=-1)
    # Jacobian correction
    jacobian_correction = torch.sum(torch.log(1.0 - clipped_action.pow(2) + 1e-6), dim=-1)
    expected_log_prob = gaussian_log_prob - jacobian_correction
    
    assert torch.allclose(log_prob, expected_log_prob)

def test_squashed_gaussian_policy():
    # Setup simple action and observation spaces
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-1e5, high=1e5, shape=(160,), dtype=np.float32)
    
    # Instantiate custom policy
    policy = SquashedGaussianActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0001
    )
    
    # Check that action distribution is indeed SquashedGaussianDistribution
    assert isinstance(policy.action_dist, SquashedGaussianDistribution)
    
    # Forward pass test
    obs = torch.randn(2, 160)
    actions, values, log_probs = policy(obs)
    
    # Verify outputs
    assert actions.shape == (2, 7)
    assert values.shape == (2, 1)
    assert log_probs.shape == (2,)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    
    # Evaluate actions
    new_values, new_log_probs, entropy = policy.evaluate_actions(obs, actions)
    assert new_values.shape == (2, 1)
    assert new_log_probs.shape == (2,)
    assert entropy.shape == (2,)
