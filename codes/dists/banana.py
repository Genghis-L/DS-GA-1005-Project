import numpy as np
from .base import BaseDistribution

class BananaDistribution(BaseDistribution):
    """Banana-shaped distribution, a common test case for MCMC."""
    
    def __init__(self, a: float = 1.0, b: float = 1.0):
        """
        Initialize banana distribution.
        
        Args:
            a: Scale parameter for x1
            b: Curvature parameter
        """
        self.a = a
        self.b = b
        
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the probability density function at x."""
        return np.exp(self.log_density(x))
        
    def log_density(self, x: np.ndarray) -> float:
        """Log of the probability density function."""
        return -0.5 * (x[0]**2 / self.a**2 + 
                      (x[1] - self.b * x[0]**2 + self.a**2)**2)
        
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log probability density function."""
        grad = np.zeros_like(x)
        grad[0] = -x[0] / self.a**2 - 2 * self.b * x[0] * (
            x[1] - self.b * x[0]**2 + self.a**2
        )
        grad[1] = -(x[1] - self.b * x[0]**2 + self.a**2)
        return grad 