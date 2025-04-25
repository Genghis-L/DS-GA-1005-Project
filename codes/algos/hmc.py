import numpy as np
from typing import Callable, List, Any, Optional

class TargetDistribution:
    """
    Base class for target distributions that can be used with both Metropolis and HMC.
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

class HMCSampler:
    """Hamiltonian Monte Carlo sampler."""
    
    def __init__(
        self,
        target: TargetDistribution,
        initial: Callable[[], np.ndarray],
        iterations: int = 10_000,
        L: int = 50,
        step_size: float = 0.1,
        mass_matrix: np.ndarray = None
    ):
        """
        Initialize the HMC sampler.
        
        Args:
            target: Target distribution implementing at least __call__.
                   For HMC, should also implement log_density and grad_log_density.
            initial: Function that returns initial position
            iterations: Number of iterations to run
            L: Number of leapfrog steps
            step_size: Size of each leapfrog step
            mass_matrix: Mass matrix M (if None, uses identity)
        """
        self.target = target
        self.initial = initial
        self.iterations = iterations
        self.L = L
        self.step_size = step_size
        self.samples = []
        
        # Check if target has required methods for HMC
        if not hasattr(target, 'grad_log_density'):
            raise ValueError("Target distribution must implement grad_log_density for HMC")
            
        # Initialize mass matrix
        self._initialize_mass_matrix(mass_matrix)
        
    def _initialize_mass_matrix(self, mass_matrix: np.ndarray = None):
        """Initialize mass matrix and its inverse."""
        if mass_matrix is None:
            # Get dimension from initial sample
            init_sample = self.initial()
            dim = len(init_sample)
            # For high dimensions, we might want to scale the mass matrix
            self.M = np.eye(dim)
            self.M_inv = np.eye(dim)
        else:
            self.M = mass_matrix
            self.M_inv = np.linalg.inv(mass_matrix)
    
    def _compute_hamiltonian(self, theta: np.ndarray, r: np.ndarray) -> float:
        """
        Compute Hamiltonian H(θ,r) = U(θ) + K(r)
        where U(θ) = -log p(θ) is potential energy
        and K(r) = r^T M^(-1) r / 2 is kinetic energy
        """
        potential = -self.target.log_density(theta)  # U(θ) = -log p(θ)
        kinetic = 0.5 * r.T @ self.M_inv @ r  # K(r) = r^T M^(-1) r / 2
        return potential + kinetic
    
    def _leapfrog(
        self,
        theta_0: np.ndarray,
        r_0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog integrator for Hamiltonian dynamics following the pseudocode.
        
        Args:
            theta_0: Initial position
            r_0: Initial momentum
            
        Returns:
            Final position and momentum after L steps
        """
        theta = theta_0.copy()
        r = r_0.copy()
        
        # Initial half step for momentum
        # Note: grad_log_density gives ∇log p(θ) which is -∇U(θ)
        r = r + (self.step_size/2) * self.target.grad_log_density(theta)
        
        # Full steps for position and momentum
        for _ in range(self.L):
            # Position update using current momentum
            theta = theta + self.step_size * self.M_inv @ r
            
            # Full momentum update if not at the end
            if _ < self.L - 1:
                r = r + self.step_size * self.target.grad_log_density(theta)
        
        # Final half step for momentum
        r = r + (self.step_size/2) * self.target.grad_log_density(theta)
        
        return theta, r
    
    def run(self) -> List[np.ndarray]:
        """
        Run the HMC algorithm following the pseudocode exactly.
        
        Returns:
            List of samples from the target distribution
        """
        self.samples = [self.initial()]
        accepted = 0
        
        for t in range(self.iterations):
            theta_t = self.samples[-1]
            
            # Sample momentum from N(0, M)
            r_t = np.random.multivariate_normal(
                mean=np.zeros_like(theta_t),
                cov=self.M
            )
            
            # Store initial state
            theta_0, r_0 = theta_t, r_t
            
            # Run leapfrog integrator
            theta_hat, r_hat = self._leapfrog(theta_0, r_0)
            
            # Metropolis-Hastings correction
            current_h = self._compute_hamiltonian(theta_0, r_0)
            proposed_h = self._compute_hamiltonian(theta_hat, r_hat)
            log_accept_ratio = current_h - proposed_h
            
            # Accept or reject
            if np.log(np.random.uniform(0, 1)) < log_accept_ratio:
                self.samples.append(theta_hat)
                accepted += 1
            else:
                self.samples.append(theta_t)
                
        print(f"HMC acceptance rate: {accepted/self.iterations:.2%}")
        return self.samples
    
    def get_samples(self) -> List[np.ndarray]:
        """Get the samples generated by the sampler."""
        return self.samples 