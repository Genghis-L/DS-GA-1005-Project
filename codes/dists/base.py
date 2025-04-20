import numpy as np
from typing import Optional

class BaseDistribution:
    """
    Base class for probability distributions that can be used with MCMC samplers.
    At minimum, must implement __call__ for density evaluation.
    Optionally can implement log_density and grad_log_density for HMC.
    """
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the probability density function at x."""
        raise NotImplementedError
        
    def log_density(self, x: np.ndarray) -> Optional[float]:
        """Log of the probability density function. Optional for HMC."""
        return np.log(self(x))
        
    def grad_log_density(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Gradient of the log probability density function. Optional for HMC."""
        return None 