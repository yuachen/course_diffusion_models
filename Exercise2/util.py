import numpy as np

import torch
from torch import nn
from torch.func import vmap, jacrev
from abc import ABC, abstractmethod


class Density(ABC):
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the log-density at x with shape (batch_size, 1).
        """
        pass

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the score function at x, i.e. the gradient of the log density
        The advantage of torch is that we can compute the score function using automatic differentiation, without having to derive it analytically
        """
        x = x.unsqueeze(1)  # (batch_size, 1, ...)
        score = vmap(jacrev(self.log_prob))(x)  # (batch_size, 1, 1, 1, ...)
        return score.squeeze((1, 2, 3))  # (batch_size, ...)


class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Return num_samples i.i.d. samples.
        """
        pass


class Gaussian(nn.Module, Density, Sampleable):
    mean : torch.Tensor
    cov : torch.Tensor
    inv_cov : torch.Tensor
    """
    Gaussian with mean and covariance.
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
        self.register_buffer("inv_cov", torch.linalg.inv(cov))
        self.dim = mean.shape[0]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.MultivariateNormal(self.mean, self.cov, validate_args=False)
        return dist.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        dist = torch.distributions.MultivariateNormal(self.mean, self.cov)
        return dist.sample((num_samples,))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return -(x - self.mean) @ self.inv_cov.T


class MixtureOfGaussians(nn.Module, Density, Sampleable):
    means: torch.Tensor
    covs: torch.Tensor
    weights: torch.Tensor
    components: nn.ModuleList
    """
    Mixture of Gaussians with given means, covariances, and weights.
    """
    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        super().__init__()
        weights = weights / weights.sum()
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)
        self.num_components = means.shape[0]
        self.dim = means.shape[1]
        self.components = nn.ModuleList(
            [Gaussian(means[i], covs[i]) for i in range(self.num_components)]
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = torch.stack(
            [
                component.log_prob(x).squeeze(-1) + torch.log(weight)
                for component, weight in zip(self.components, self.weights)
            ],
            dim=1,
        )
        return torch.logsumexp(log_probs, dim=1, keepdim=True)

    def sample(self, num_samples: int) -> torch.Tensor:
        component_ids = torch.multinomial(self.weights, num_samples, replacement=True)
        samples = torch.empty(num_samples, self.dim, device=self.means.device, dtype=self.means.dtype)

        for k in range(self.num_components):
            mask = component_ids == k
            count = int(mask.sum().item())
            if count > 0:
                samples[mask] = self.components[k].sample(count)
        return samples

    def score(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = torch.stack(
            [
                component.log_prob(x).squeeze(-1) + torch.log(weight)
                for component, weight in zip(self.components, self.weights)
            ],
            dim=1,
        )
        log_normalizer = torch.logsumexp(log_probs, dim=1, keepdim=True)
        responsibilities = torch.exp(log_probs - log_normalizer)  # (batch_size, num_components)

        component_scores = torch.stack(
            [component.score(x) for component in self.components], dim=1
        )  # (batch_size, num_components, dim)

        return (responsibilities.unsqueeze(-1) * component_scores).sum(dim=1)

    @classmethod
    def two_modes_2D(
        cls,
        distance: float = 8.0,
        cov_scale: float = 1.0,
        weights: torch.Tensor | None = None,
    ) -> "MixtureOfGaussians":
        means = torch.tensor(
            [[-distance / 2, 0.0], [distance / 2, 0.0]],
            dtype=torch.float32,
        )
        covs = cov_scale * torch.eye(2).unsqueeze(0).repeat(2, 1, 1)
        if weights is None:
            weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
        return cls(means, covs, weights)



class SDE(ABC):
    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        for i in range(len(ts) - 1):
            dt = ts[i + 1] - ts[i]
            x = self.step(x, ts[i], dt)
        return x

    @torch.no_grad()
    def simulate_trajectory(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        trajectory = [x.clone()]
        for i in range(len(ts) - 1):
            dt = ts[i + 1] - ts[i]
            x = self.step(x, ts[i], dt)
            trajectory.append(x.clone())
        return torch.stack(trajectory, dim=1)


class EulerMaruyama(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        drift = self.sde.drift(x, t)
        diffusion = self.sde.diffusion(x, t)
        noise = torch.randn_like(x)
        return x + drift * dt + diffusion * noise * torch.sqrt(dt)


class OrnsteinUhlenbeck(SDE):
    def __init__(self, theta: float = 1.0, sigma: float = np.sqrt(2.0)):
        self.theta = theta
        self.sigma = sigma

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -self.theta * x

    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.ones_like(x)
