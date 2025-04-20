import numpy as np
from .base import BaseDistribution
from .normal import NormalDistribution

class MixtureDistribution(BaseDistribution):
    """Mixture of Gaussian distributions."""
    
    def __init__(
        self,
        means: list[np.ndarray],
        covs: list[np.ndarray],
        weights: list[float] = None
    ):
        """
        Initialize mixture distribution.
        
        Args:
            means: List of mean vectors for each component
            covs: List of covariance matrices for each component
            weights: Mixing weights. If None, defaults to uniform weights.
        """
        self.components = [
            NormalDistribution(mean, cov)
            for mean, cov in zip(means, covs)
        ]
        self.weights = (
            np.array(weights) if weights is not None
            else np.ones(len(means)) / len(means)
        )
        
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the probability density function at x."""
        return np.sum([
            w * component(x)
            for w, component in zip(self.weights, self.components)
        ])
        
    def log_density(self, x: np.ndarray) -> float:
        """Log of the probability density function."""
        return np.log(self(x))
        
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log probability density function."""
        # Compute weighted sum of component densities
        densities = np.array([component(x) for component in self.components])
        weighted_densities = self.weights * densities
        total_density = np.sum(weighted_densities)
        
        # Compute weighted sum of component gradients
        gradients = np.array([
            component.grad_log_density(x)
            for component in self.components
        ])
        
        # Combine using chain rule
        return np.sum(
            weighted_densities[:, np.newaxis] * gradients,
            axis=0
        ) / total_density 