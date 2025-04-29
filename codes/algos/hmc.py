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
        mass_matrix: np.ndarray = None,
        kinetic_energy: str = "gaussian",
        kinetic_params: Optional[dict] = None
    ):
        """
        Initialize the HMC sampler.
        
        Args:
            target: Target distribution implementing at least __call__.
                   For HMC, should also implement log_density and grad_log_density.
            initial: Function that returns initial position
            iterations: Number of iterations to run
            L: Number of leapfrog steps
            step_size: Size of each leapfrog step, epsilon in the paper
            mass_matrix: Mass matrix M (if None, uses identity)
            kinetic_energy: Type of kinetic energy distribution to use
                          ('gaussian', 'gaussian_ars', 'student_t', 'relativistic', 'uniform')
            kinetic_params: Parameters for the kinetic energy distribution
                          - For 'student_t': {'nu': float} (degrees of freedom)
                          - For 'relativistic': {'mass': float} (particle mass)
                          - For 'uniform': {'scale': float} (scale of uniform distribution)
        """
        self.target = target
        self.initial = initial
        self.iterations = iterations
        self.L = L
        self.step_size = step_size  # leapfrog stepsize, ε
        self.samples = []
        self.acceptance_rate = 0.0  # Track acceptance rate
        self.kinetic_energy = kinetic_energy
        self.kinetic_params = kinetic_params or {}
        
        # Validate kinetic energy parameters
        if kinetic_energy == "student_t":
            if "nu" not in self.kinetic_params:
                raise ValueError("Student's t kinetic energy requires 'nu' parameter")
            if self.kinetic_params["nu"] <= 0:
                raise ValueError("Degrees of freedom 'nu' must be positive")
        elif kinetic_energy == "relativistic":
            if "mass" not in self.kinetic_params:
                raise ValueError("Relativistic kinetic energy requires 'mass' parameter")
            if self.kinetic_params["mass"] <= 0:
                raise ValueError("Mass must be positive")
        elif kinetic_energy == "uniform":
            if "scale" not in self.kinetic_params:
                raise ValueError("Uniform kinetic energy requires 'scale' parameter")
            if self.kinetic_params["scale"] <= 0:
                raise ValueError("Scale must be positive")
            
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
    
    def _compute_kinetic_energy(self, r: np.ndarray) -> float:
        """Compute kinetic energy based on the chosen distribution."""
        if self.kinetic_energy == "gaussian":
            return 0.5 * r.T @ self.M_inv @ r
        elif self.kinetic_energy == "gaussian_ars":
            # Gaussian with ARS (Adaptive Rejection Sampling)
            return 0.5 * r.T @ self.M_inv @ r
        elif self.kinetic_energy == "student_t":
            nu = self.kinetic_params["nu"]
            return (nu / 2) * np.log(1 + r.T @ self.M_inv @ r / nu)
        elif self.kinetic_energy == "relativistic":
            m = self.kinetic_params["mass"]
            return np.sqrt(r.T @ self.M_inv @ r + m * m) - m
        elif self.kinetic_energy == "uniform":
            scale = self.kinetic_params["scale"]
            # For uniform distribution, kinetic energy is constant within the support
            # and infinite outside
            if np.all(np.abs(r) <= scale):
                return 0.0
            else:
                return float('inf')
        else:
            raise ValueError(f"Unknown kinetic energy type: {self.kinetic_energy}")
    
    def _compute_kinetic_gradient(self, r: np.ndarray) -> np.ndarray:
        """Compute gradient of kinetic energy with respect to momentum."""
        if self.kinetic_energy == "gaussian":
            return self.M_inv @ r
        elif self.kinetic_energy == "gaussian_ars":
            return self.M_inv @ r
        elif self.kinetic_energy == "student_t":
            nu = self.kinetic_params["nu"]
            return self.M_inv @ r / (1 + r.T @ self.M_inv @ r / nu)
        elif self.kinetic_energy == "relativistic":
            m = self.kinetic_params["mass"]
            return self.M_inv @ r / np.sqrt(r.T @ self.M_inv @ r + m * m)
        elif self.kinetic_energy == "uniform":
            scale = self.kinetic_params["scale"]
            # For uniform distribution, gradient is zero within support
            # and undefined outside
            if np.all(np.abs(r) <= scale):
                return np.zeros_like(r)
            else:
                raise ValueError("Momentum outside uniform support")
        else:
            raise ValueError(f"Unknown kinetic energy type: {self.kinetic_energy}")
    
    def _sample_momentum(self, dim: int) -> np.ndarray:
        """Sample momentum from the chosen kinetic energy distribution."""
        if self.kinetic_energy == "gaussian":
            return np.random.multivariate_normal(
                mean=np.zeros(dim),
                cov=self.M
            )
        elif self.kinetic_energy == "gaussian_ars":
            # Gaussian with ARS (Adaptive Rejection Sampling)
            return np.random.multivariate_normal(
                mean=np.zeros(dim),
                cov=self.M
            )
        elif self.kinetic_energy == "student_t":
            nu = self.kinetic_params["nu"]
            return np.random.standard_t(nu, size=dim) * np.sqrt(np.diag(self.M))
        elif self.kinetic_energy == "relativistic":
            m = self.kinetic_params["mass"]
            # Sample from relativistic distribution using rejection sampling
            while True:
                r = np.random.multivariate_normal(
                    mean=np.zeros(dim),
                    cov=self.M
                )
                if np.random.random() < np.exp(-self._compute_kinetic_energy(r)):
                    return r
        elif self.kinetic_energy == "uniform":
            scale = self.kinetic_params["scale"]
            # Sample from uniform distribution
            return np.random.uniform(-scale, scale, size=dim)
        else:
            raise ValueError(f"Unknown kinetic energy type: {self.kinetic_energy}")
    
    def _compute_hamiltonian(self, theta: np.ndarray, r: np.ndarray) -> float:
        """
        Compute Hamiltonian H(θ,r) = U(θ) + K(r)
        where U(θ) = -log p(θ) is potential energy
        and K(r) is the kinetic energy based on the chosen distribution
        """
        potential = -self.target.log_density(theta)  # U(θ) = -log p(θ)
        kinetic = self._compute_kinetic_energy(r)
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
        r = r + (self.step_size/2) * self.target.grad_log_density(theta)
        
        # Full steps for position and momentum
        for _ in range(self.L):
            # Position update using current momentum
            theta = theta + self.step_size * self._compute_kinetic_gradient(r)
            
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
            
            # Sample momentum from the chosen distribution
            r_t = self._sample_momentum(len(theta_t))
            
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
        
        # Store acceptance rate
        self.acceptance_rate = accepted / self.iterations
        print(f"HMC acceptance rate: {self.acceptance_rate:.2%}")
        return self.samples
    
    def get_samples(self) -> List[np.ndarray]:
        """Get the samples generated by the sampler."""
        return self.samples
        
    def get_acceptance_rate(self) -> float:
        """Get the acceptance rate from the last run."""
        return self.acceptance_rate 