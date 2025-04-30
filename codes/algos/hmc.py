import numpy as np
from typing import Callable, List, Any, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        step_size: float = 0.1,
        trajectory_length: float = 0.5,  # s = ε * L
        mass_matrix: np.ndarray = None,
        kinetic_energy: str = "gaussian",
        kinetic_params: Optional[dict] = None,
        use_metropolis: bool = True
    ):
        """
        Initialize the HMC sampler.
        
        Args:
            target: Target distribution implementing at least __call__.
                   For HMC, should also implement log_density and grad_log_density.
            initial: Function that returns initial position
            iterations: Number of iterations to run
            step_size: Size of each leapfrog step (ε)
            trajectory_length: Total trajectory length (s = ε * L)
            mass_matrix: Mass matrix M (if None, uses identity)
            kinetic_energy: Type of kinetic energy distribution to use
                          ('gaussian', 'student_t', 'alpha_norm')
            kinetic_params: Parameters for the kinetic energy distribution
                          - For 'student_t': {'nu': float} (degrees of freedom)
                          - For 'alpha_norm': {'alpha': float} (norm power)
            use_metropolis: Whether to use Metropolis correction step (default: True)
        """
        self.target = target
        self.initial = initial
        self.iterations = iterations
        self.step_size = step_size  # ε
        self.trajectory_length = trajectory_length  # s = ε * L
        self.L = int(trajectory_length // step_size) + 1  # Number of leapfrog steps
        self.samples = []
        self.acceptance_rate = 0.0  # Track acceptance rate
        self.kinetic_energy = kinetic_energy
        self.kinetic_params = kinetic_params or {}
        self.use_metropolis = use_metropolis
        
        # Validate kinetic energy parameters
        if kinetic_energy == "student_t":
            if "nu" not in self.kinetic_params:
                raise ValueError("Student's t kinetic energy requires 'nu' parameter")
            if self.kinetic_params["nu"] <= 0:
                raise ValueError("Degrees of freedom 'nu' must be positive")
        elif kinetic_energy == "alpha_norm":
            if "alpha" not in self.kinetic_params:
                raise ValueError("Alpha-norm kinetic energy requires 'alpha' parameter")
            if self.kinetic_params["alpha"] <= 0:
                raise ValueError("Alpha parameter must be positive")
            self.alpha = self.kinetic_params["alpha"]
            
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
        elif self.kinetic_energy == "student_t":
            nu = self.kinetic_params["nu"]
            return (nu / 2) * np.log(1 + r.T @ self.M_inv @ r / nu)
        elif self.kinetic_energy == "alpha_norm":
            # K(p) = ||M^(-1/2)p||_α / α
            M_inv_sqrt_p = np.linalg.cholesky(self.M_inv) @ r
            return np.sum(np.abs(M_inv_sqrt_p) ** self.alpha) / self.alpha
        else:
            raise ValueError(f"Unknown kinetic energy type: {self.kinetic_energy}")
    
    def _compute_kinetic_gradient(self, r: np.ndarray) -> np.ndarray:
        """Compute gradient of kinetic energy with respect to momentum."""
        if self.kinetic_energy == "gaussian":
            return self.M_inv @ r
        elif self.kinetic_energy == "student_t":
            nu = self.kinetic_params["nu"]
            return self.M_inv @ r / (1 + r.T @ self.M_inv @ r / nu)
        elif self.kinetic_energy == "alpha_norm":
            # ∇K(p) = M^(-1/2) * sign(M^(-1/2)p) * |M^(-1/2)p|^(α-1)
            M_inv_sqrt = np.linalg.cholesky(self.M_inv)
            M_inv_sqrt_p = M_inv_sqrt @ r
            return M_inv_sqrt @ (np.sign(M_inv_sqrt_p) * np.abs(M_inv_sqrt_p) ** (self.alpha - 1))
        else:
            raise ValueError(f"Unknown kinetic energy type: {self.kinetic_energy}")
    
    def _sample_momentum(self, dim: int) -> np.ndarray:
        """Sample momentum from the chosen kinetic energy distribution."""
        if self.kinetic_energy == "gaussian":
            return np.random.multivariate_normal(
                mean=np.zeros(dim),
                cov=self.M
            )
        elif self.kinetic_energy == "student_t":
            nu = self.kinetic_params["nu"]
            return np.random.standard_t(nu, size=dim) * np.sqrt(np.diag(self.M))
        elif self.kinetic_energy == "alpha_norm":
            # For α-norm, we use rejection sampling
            while True:
                r = np.random.multivariate_normal(
                    mean=np.zeros(dim),
                    cov=self.M
                )
                if np.random.random() < np.exp(-self._compute_kinetic_energy(r)):
                    return r
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
            
            if self.use_metropolis:
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
            else:
                # Always accept the proposal without Metropolis correction
                self.samples.append(theta_hat)
                accepted += 1
        
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
    

class HMCWithIntegrator(HMCSampler):
    """HMC sampler with configurable integrator."""
    
    def __init__(
        self,
        target: TargetDistribution,
        initial: callable,
        integrator: str = 'leapfrog',
        iterations: int = 10_000,
        step_size: float = 0.1,
        trajectory_length: float = 0.5,  # s = ε * L
        mass_matrix: np.ndarray = None,
        kinetic_energy: str = "gaussian",
        kinetic_params: Optional[dict] = None,
        use_metropolis: bool = True
    ):
        super().__init__(
            target=target,
            initial=initial,
            iterations=iterations,
            step_size=step_size,
            trajectory_length=trajectory_length,
            mass_matrix=mass_matrix,
            kinetic_energy=kinetic_energy,
            kinetic_params=kinetic_params,
            use_metropolis=use_metropolis
        )
        self.integrator = integrator
        self.acceptance_rate = 0.0  # Initialize acceptance rate
        self.flop_count = 0  # Initialize FLOP counter
        
    def get_flop_count(self):
        return self.flop_count
    
    def _euler_integrator(
        self,
        theta_0: np.ndarray,
        r_0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simple Euler integrator."""
        theta = theta_0.copy()
        r = r_0.copy()
        
        for _ in range(self.L):
            # Update position and momentum using current gradients
            theta = theta + self.step_size * self.M_inv @ r
            r = r + self.step_size * self.target.grad_log_density(theta)
            
        return theta, r
        
    def _modified_euler_integrator(
        self,
        theta_0: np.ndarray,
        r_0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Modified Euler integrator following equations 5.16-5.17."""
        theta = theta_0.copy()
        r = r_0.copy()
        
        for _ in range(self.L):
            # First update momentum using current position
            r = r + self.step_size * self.target.grad_log_density(theta)
            # Then update position using updated momentum
            theta = theta + self.step_size * self.M_inv @ r
            
        return theta, r
    
    def _leapfrog(
        self,
        theta_0: np.ndarray,
        r_0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standard leapfrog integrator with FLOP counting."""
        r = r_0.copy()
        theta = theta_0.copy()
        d = len(theta)
        # Estimate FLOPs for vector addition (d), scalar multiplication (d), and gradient (assume 2d)
        # Initial half step for momentum
        r = r + (self.step_size/2) * self.target.grad_log_density(theta)
        self.flop_count += 2 * d + 2 * d  # grad_log_density (2d), scalar mult (d), add (d)
        # Full steps for position and momentum
        for _ in range(self.L):
            # Position update
            theta = theta + self.step_size * self.M_inv @ r
            self.flop_count += d * d + d + d  # matmul (d*d), scalar mult (d), add (d)
            # Momentum update (except at end)
            if _ < self.L - 1:
                grad = self.target.grad_log_density(theta)
                r = r + self.step_size * grad
                self.flop_count += 2 * d + d  # grad_log_density (2d), scalar mult (d), add (d)
        # Final half step for momentum
        grad = self.target.grad_log_density(theta)
        r = r + (self.step_size/2) * grad
        self.flop_count += 2 * d + d  # grad_log_density (2d), scalar mult (d), add (d)
        return theta, r
    
    def _integrate(
        self,
        theta_0: np.ndarray,
        r_0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose integrator based on configuration."""
        if self.integrator == 'euler':
            return self._euler_integrator(theta_0, r_0)
        elif self.integrator == 'modified_euler':
            return self._modified_euler_integrator(theta_0, r_0)
        else:  # leapfrog
            return self._leapfrog(theta_0, r_0)
            
    def run(self) -> list:
        """Run HMC with chosen integrator."""
        self.samples = [self.initial()]
        accepted = 0
        
        for _ in range(self.iterations):
            theta_t = self.samples[-1]
            
            # Sample momentum from the chosen distribution
            r_t = self._sample_momentum(len(theta_t))
            
            # Store initial state
            theta_0, r_0 = theta_t, r_t
            
            # Run leapfrog integrator
            theta_hat, r_hat = self._integrate(theta_0, r_0)
            
            if self.use_metropolis:
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
            else:
                # Always accept the proposal without Metropolis correction
                self.samples.append(theta_hat)
                accepted += 1
        
        # Store acceptance rate
        self.acceptance_rate = accepted / self.iterations
        print(f"HMC acceptance rate: {self.acceptance_rate:.2%}")
        return self.samples
    

def compute_energy_error(sampler: HMCWithIntegrator, n_test: int = 100) -> float:
    """Compute average relative energy error for the integrator."""
    errors = []
    theta = sampler.initial()
    
    for _ in range(n_test):
        # Sample momentum
        r = np.random.multivariate_normal(
            mean=np.zeros_like(theta),
            cov=sampler.M
        )
        
        # Store initial energy
        initial_energy = sampler._compute_hamiltonian(theta, r)
        
        # Run integrator
        theta_new, r_new = sampler._integrate(theta, r)
        
        # Compute final energy
        final_energy = sampler._compute_hamiltonian(theta_new, r_new)
        
        # Compute relative error
        error = abs((final_energy - initial_energy) / initial_energy)
        errors.append(error)
        
        # Update position for next test
        theta = theta_new
        
    return np.mean(errors)

def compute_relative_energy_error(sampler: HMCWithIntegrator, n_test: int = 100) -> float:
    """Compute mean relative energy error for the sampler over n_test trajectories."""
    errors = []
    theta = sampler.initial()
    for _ in range(n_test):
        r = np.random.multivariate_normal(mean=np.zeros_like(theta), cov=sampler.M)
        initial_energy = sampler._compute_hamiltonian(theta, r)
        theta_new, r_new = sampler._integrate(theta, r)
        final_energy = sampler._compute_hamiltonian(theta_new, r_new)
        rel_error = abs(final_energy - initial_energy) / (abs(initial_energy) + 1e-12)
        errors.append(rel_error)
        theta = theta_new
    return np.mean(errors)