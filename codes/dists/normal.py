import numpy as np
from .base import BaseDistribution

class NormalDistribution(BaseDistribution):
    """Multivariate normal distribution."""
    
    def __init__(self, mean: np.ndarray = None, cov: np.ndarray = None):
        """
        Initialize normal distribution.
        
        Args:
            mean: Mean vector. If None, defaults to zero vector.
            cov: Covariance matrix. If None, defaults to identity matrix.
        """
        self.mean = mean if mean is not None else np.zeros(1)
        self.cov = cov if cov is not None else np.eye(len(self.mean))
        self.inv_cov = np.linalg.inv(self.cov)
        self.dim = len(self.mean)
        
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the probability density function at x."""
        diff = x - self.mean
        return np.exp(-0.5 * diff.T @ self.inv_cov @ diff) / np.sqrt(
            (2 * np.pi) ** self.dim * np.linalg.det(self.cov)
        )
        
    def log_density(self, x: np.ndarray) -> float:
        """Log of the probability density function."""
        diff = x - self.mean
        return -0.5 * (diff.T @ self.inv_cov @ diff + 
                      np.log((2 * np.pi) ** self.dim * np.linalg.det(self.cov)))
        
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log probability density function."""
        return -self.inv_cov @ (x - self.mean) 