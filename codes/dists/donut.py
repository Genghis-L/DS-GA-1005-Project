from algos.hmc import TargetDistribution
import numpy as np
class DonutDistribution(TargetDistribution):
    """N-dimensional shell distribution (generalized donut)."""
    
    def __init__(self, radius: float = 3.0, sigma2: float = 0.05, dim: int = 2):
        """
        Initialize N-dimensional shell distribution.
        
        Args:
            radius: Target radius of the shell
            sigma2: Variance parameter controlling shell thickness
            dim: Number of dimensions
        """
        self.radius = radius
        self.sigma2 = sigma2
        self.dim = dim
        
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate unnormalized density."""
        r = np.linalg.norm(x)
        return np.exp(-(r - self.radius) ** 2 / self.sigma2)
    
    def log_density(self, x: np.ndarray) -> float:
        """Log of unnormalized density."""
        r = np.linalg.norm(x)
        return -(r - self.radius) ** 2 / self.sigma2
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """Gradient of log density."""
        r = np.linalg.norm(x)
        if r == 0:
            return np.zeros_like(x)
        return 2 * x * (self.radius / r - 1) / self.sigma2