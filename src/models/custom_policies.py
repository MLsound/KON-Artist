import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

class SquashedGaussianDistribution(DiagGaussianDistribution):
    """
    Diagonal Gaussian distribution squashed to [-1, 1] using tanh.
    Applies change-of-variables log-prob corrections dynamically.
    """
    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.gaussian_dist = None

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "SquashedGaussianDistribution":
        """
        Instantiates the raw Gaussian distribution.
        """
        std = torch.exp(log_std)
        self.gaussian_dist = Normal(mean_actions, std)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes log-prob of squashed actions using the change-of-variables correction.
        """
        # Inverse mapping: x = arctanh(a)
        # Clip to avoid division by zero or log of negative
        clipped_actions = torch.clamp(actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        raw_actions = 0.5 * torch.log((1.0 + clipped_actions) / (1.0 - clipped_actions))
        
        # Base gaussian log prob of raw action
        gaussian_log_prob = self.gaussian_dist.log_prob(raw_actions).sum(dim=-1)
        
        # Jacobian determinant correction
        jacobian_correction = torch.sum(torch.log(1.0 - clipped_actions.pow(2) + 1e-6), dim=-1)
        return gaussian_log_prob - jacobian_correction

    def entropy(self) -> torch.Tensor:
        """
        Returns raw Gaussian entropy as a heuristic proxy.
        """
        return self.gaussian_dist.entropy().sum(dim=-1)

    def sample(self) -> torch.Tensor:
        """
        Samples from Gaussian and applies tanh.
        """
        raw_actions = self.gaussian_dist.rsample()
        return torch.tanh(raw_actions)

    def mode(self) -> torch.Tensor:
        """
        Returns squashed mean.
        """
        return torch.tanh(self.gaussian_dist.mean)

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(mean_actions, log_std)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedGaussianActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy utilizing the SquashedGaussianDistribution for continuous environments.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # Swap the distribution before setup in super()._build
        self.action_dist = SquashedGaussianDistribution(self.action_space.shape[0])
        super()._build(lr_schedule)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> SquashedGaussianDistribution:
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std.expand_as(mean_actions)
        return self.action_dist.proba_distribution(mean_actions, log_std)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor) -> SquashedGaussianDistribution:
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def actions_from_distribution(self, action_dist: SquashedGaussianDistribution, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return action_dist.mode()
        return action_dist.sample()
