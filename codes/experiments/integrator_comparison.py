import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.hmc import HMCSampler, TargetDistribution
from dists.normal import NormalDistribution

class HMCWithIntegrator(HMCSampler):
    """HMC sampler with configurable integrator."""
    
    def __init__(
        self,
        target: TargetDistribution,
        initial: callable,
        integrator: str = 'leapfrog',
        iterations: int = 10_000,
        L: int = 50,
        step_size: float = 0.1,
        mass_matrix: np.ndarray = None
    ):
        super().__init__(target, initial, iterations, L, step_size, mass_matrix)
        self.integrator = integrator
        
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
        """Standard leapfrog integrator."""
        r = r_0.copy()
        theta = theta_0.copy()
        
        # Initial half step for momentum
        r = r + (self.step_size/2) * self.target.grad_log_density(theta)
        
        # Full steps for position and momentum
        for _ in range(self.L):
            # Position update
            theta = theta + self.step_size * self.M_inv @ r
            # Momentum update (except at end)
            if _ < self.L - 1:
                r = r + self.step_size * self.target.grad_log_density(theta)
        
        # Final half step for momentum
        r = r + (self.step_size/2) * self.target.grad_log_density(theta)
        
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
        
        for t in range(self.iterations):
            theta_t = self.samples[-1]
            
            # Sample momentum from N(0, M)
            r_t = np.random.multivariate_normal(
                mean=np.zeros_like(theta_t),
                cov=self.M
            )
            
            # Store initial state
            theta_0, r_0 = theta_t, r_t
            
            # Run integrator
            theta_hat, r_hat = self._integrate(theta_0, r_0)
            
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
                
        print(f"HMC acceptance rate ({self.integrator}): {accepted/self.iterations:.2%}")
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

def run_comparison(
    n_samples: int = 1000,
    dim: int = 2,
    target_std: float = 1.0,
    step_sizes: list = [0.01, 0.05, 0.1, 0.2, 0.5],
    n_leapfrog_steps: int = 20,
    output_file: str = None
):
    """Run comparison of different HMC integrators."""
    # Setup target distribution (standard normal)
    target = NormalDistribution(
        mean=np.zeros(dim),
        cov=target_std**2 * np.eye(dim)
    )
    initial = lambda: np.zeros(dim)
    
    integrators = ['euler', 'modified_euler', 'leapfrog']
    results = []
    
    for integrator in integrators:
        for step_size in step_sizes:
            # Run sampler
            start_time = time.time()
            sampler = HMCWithIntegrator(
                target=target,
                initial=initial,
                integrator=integrator,
                iterations=n_samples,
                L=n_leapfrog_steps,
                step_size=step_size
            )
            samples = np.array(sampler.run())
            end_time = time.time()
            
            # Compute metrics
            accept_rate = len(np.unique(samples, axis=0)) / len(samples)
            time_per_sample = (end_time - start_time) / n_samples
            energy_error = compute_energy_error(sampler)
            
            results.append({
                'integrator': integrator,
                'step_size': step_size,
                'accept_rate': accept_rate,
                'time_per_sample': time_per_sample,
                'energy_error': energy_error
            })
            
            print(f"\nResults for {integrator.upper()} (step_size={step_size}):")
            print(f"  Acceptance rate: {accept_rate:.2%}")
            print(f"  Time per sample: {time_per_sample*1e6:.2f} μs")
            print(f"  Energy error: {energy_error:.2e}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"HMC Integrator Comparison ({dim}D)")
    
    # Group results by integrator
    for integrator in integrators:
        int_results = [r for r in results if r['integrator'] == integrator]
        step_sizes = [r['step_size'] for r in int_results]
        accept_rates = [r['accept_rate'] for r in int_results]
        times = [r['time_per_sample'] for r in int_results]
        errors = [r['energy_error'] for r in int_results]
        
        # Plot acceptance rate
        axes[0].plot(step_sizes, accept_rates, 'o-', label=integrator)
        axes[0].set_xlabel('Step Size')
        axes[0].set_ylabel('Acceptance Rate')
        axes[0].set_title('Acceptance Rate vs Step Size')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot time per sample
        axes[1].plot(step_sizes, np.array(times)*1e6, 'o-', label=integrator)
        axes[1].set_xlabel('Step Size')
        axes[1].set_ylabel('Time per Sample (μs)')
        axes[1].set_title('Computation Time vs Step Size')
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot energy error
        axes[2].plot(step_sizes, errors, 'o-', label=integrator)
        axes[2].set_xlabel('Step Size')
        axes[2].set_ylabel('Relative Energy Error')
        axes[2].set_title('Energy Conservation Error')
        axes[2].set_yscale('log')
        axes[2].grid(True)
        axes[2].legend()
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare different HMC integrators")
    parser.add_argument("--dim", type=int, default=2,
                       help="Number of dimensions (default: 2)")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--target-std", type=float, default=1.0,
                       help="Standard deviation of target normal distribution (default: 1.0)")
    parser.add_argument("--step-sizes", type=float, nargs='+',
                       default=[0.01, 0.05, 0.1, 0.2, 0.5],
                       help="Step sizes to test (default: 0.01 0.05 0.1 0.2 0.5)")
    parser.add_argument("--n-leapfrog", type=int, default=20,
                       help="Number of integration steps (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file name (default: integrator_comparison_{dim}d.png)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'integrator_comparison_{args.dim}d.png'
    
    run_comparison(
        n_samples=args.n_samples,
        dim=args.dim,
        target_std=args.target_std,
        step_sizes=args.step_sizes,
        n_leapfrog_steps=args.n_leapfrog,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 