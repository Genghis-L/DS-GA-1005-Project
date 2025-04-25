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
from dists.donut import DonutDistribution

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
        self.acceptance_rate = 0.0  # Initialize acceptance rate
        
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
        
        # Store acceptance rate
        self.acceptance_rate = accepted / self.iterations
        print(f"HMC acceptance rate ({self.integrator}): {self.acceptance_rate:.2%}")
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
    target_dist: str = 'normal',
    target_std: float = 1.0,
    donut_radius: float = 3.0,
    donut_sigma2: float = 0.05,
    step_sizes: list = [0.1, 0.5, 1.0],
    n_leapfrog_steps: int = 20,
    output_file: str = None
):
    """Run comparison of different HMC integrators."""
    # Setup target distribution
    if target_dist == 'normal':
        target = NormalDistribution(
            mean=np.zeros(dim),
            cov=target_std**2 * np.eye(dim)
        )
        initial = lambda: np.zeros(dim)
        plot_range = (-3*target_std, 3*target_std)
    else:  # donut
        target = DonutDistribution(
            radius=donut_radius,
            sigma2=donut_sigma2,
            dim=dim
        )
        initial = lambda: np.array([donut_radius] + [0.0] * (dim-1))  # Start on x-axis at radius
        plot_range = (-4, 4)  # Fixed range for donut visualization
    
    integrators = ['euler', 'modified_euler', 'leapfrog']
    results = []
    
    # Create figure for metrics comparison
    fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(15, 5))
    fig_metrics.suptitle(f"HMC Integrator Comparison - {target_dist.title()} Distribution ({dim}D)")
    
    # Create figure for sample visualization (one row per step size)
    n_step_sizes = len(step_sizes)
    fig_samples, axes_samples = plt.subplots(n_step_sizes, 3, figsize=(15, 5*n_step_sizes))
    fig_samples.suptitle(f"HMC Integrator Sample Comparison - {target_dist.title()} Distribution ({dim}D)")
    
    if n_step_sizes == 1:
        axes_samples = axes_samples.reshape(1, -1)
    
    # Plot settings for sample visualization
    x = np.linspace(plot_range[0], plot_range[1], 100)
    y = np.linspace(plot_range[0], plot_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = target(np.array([X[j, i], Y[j, i]]))
    
    # Run comparison for each integrator and step size
    for step_idx, step_size in enumerate(step_sizes):
        for int_idx, integrator in enumerate(integrators):
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
            time_per_sample = (end_time - start_time) / n_samples
            energy_error = compute_energy_error(sampler)
            
            results.append({
                'integrator': integrator,
                'step_size': step_size,
                'acc_rate': sampler.get_acceptance_rate(),
                'time_per_sample': time_per_sample,
                'energy_error': energy_error
            })
            
            print(f"\nResults for {integrator.upper()} (step_size={step_size}):")
            print(f"  Acceptance rate: {sampler.get_acceptance_rate():.2%}")
            print(f"  Time per sample: {time_per_sample*1e6:.2f} μs")
            print(f"  Energy error: {energy_error:.2e}")
            
            # Plot samples for 2D case
            if dim == 2:
                ax = axes_samples[step_idx, int_idx]
                # Plot target density contours
                ax.contour(X, Y, Z, levels=10, colors='r', alpha=0.3)
                # Plot samples
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
                ax.set_title(f"{integrator.title()}\n(ε={step_size}, L={n_leapfrog_steps})")
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
                # Add statistics as text
                stat_text = (f"Acc. Rate: {sampler.get_acceptance_rate():.2%}\n"
                           f"Energy Error: {energy_error:.2e}")
                ax.text(0.02, 0.98, stat_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Set consistent axis limits
                ax.set_xlim(plot_range)
                ax.set_ylim(plot_range)
    
    # Group results by integrator for metric plots
    for integrator in integrators:
        int_results = [r for r in results if r['integrator'] == integrator]
        step_sizes_plot = [r['step_size'] for r in int_results]
        acc_rates = [r['acc_rate'] for r in int_results]
        times = [r['time_per_sample'] for r in int_results]
        errors = [r['energy_error'] for r in int_results]
        
        # Plot acceptance rate
        axes_metrics[0].plot(step_sizes_plot, acc_rates, 'o-', label=integrator)
        axes_metrics[0].set_xlabel('Step Size')
        axes_metrics[0].set_ylabel('Acceptance Rate')
        axes_metrics[0].set_title('Acceptance Rate vs Step Size')
        axes_metrics[0].grid(True)
        axes_metrics[0].legend()
        
        # Plot time per sample
        axes_metrics[1].plot(step_sizes_plot, np.array(times)*1e6, 'o-', label=integrator)
        axes_metrics[1].set_xlabel('Step Size')
        axes_metrics[1].set_ylabel('Time per Sample (μs)')
        axes_metrics[1].set_title('Computation Time vs Step Size')
        axes_metrics[1].grid(True)
        axes_metrics[1].legend()
        
        # Plot energy error
        axes_metrics[2].plot(step_sizes_plot, errors, 'o-', label=integrator)
        axes_metrics[2].set_xlabel('Step Size')
        axes_metrics[2].set_ylabel('Relative Energy Error')
        axes_metrics[2].set_title('Energy Conservation Error')
        axes_metrics[2].set_yscale('log')
        axes_metrics[2].grid(True)
        axes_metrics[2].legend()
    
    # Save plots
    if output_file:
        base_name = output_file.rsplit('.', 1)[0]
        fig_metrics.savefig(f"{base_name}_metrics.png", dpi=300, bbox_inches='tight')
        if dim == 2:
            fig_samples.savefig(f"{base_name}_samples.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Compare different HMC integrators")
    parser.add_argument("--distribution", choices=['normal', 'donut'], default='normal',
                       help="Target distribution (default: normal)")
    parser.add_argument("--dim", type=int, default=2,
                       help="Number of dimensions (default: 2)")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--target-std", type=float, default=1.0,
                       help="Standard deviation of target normal distribution (default: 1.0)")
    parser.add_argument("--donut-radius", type=float, default=3.0,
                       help="Target radius for donut distribution (default: 3.0)")
    parser.add_argument("--donut-sigma2", type=float, default=0.05,
                       help="Shell thickness for donut distribution (default: 0.05)")
    parser.add_argument("--step-sizes", type=float, nargs='+',
                       default=[0.1, 0.5, 1.0],
                       help="Step sizes to test")
    parser.add_argument("--n-leapfrog", type=int, default=20,
                       help="Number of integration steps (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file name (default: integrator_comparison_{dist}_{dim}d.png)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'integrator_comparison_{args.distribution}_{args.dim}d.png'
    
    print(f"\nRunning comparison for {args.distribution} distribution in {args.dim} dimensions...")
    print(f"Parameters:")
    if args.distribution == 'normal':
        print(f"  - Target std: {args.target_std}")
    else:
        print(f"  - Donut radius: {args.donut_radius}")
        print(f"  - Shell thickness: {args.donut_sigma2}")
    print(f"  - Number of samples: {args.n_samples}")
    print(f"  - Step sizes: {args.step_sizes}")
    print(f"  - Leapfrog steps: {args.n_leapfrog}")
    print(f"  - Output file: {args.output}\n")
    
    run_comparison(
        n_samples=args.n_samples,
        dim=args.dim,
        target_dist=args.distribution,
        target_std=args.target_std,
        donut_radius=args.donut_radius,
        donut_sigma2=args.donut_sigma2,
        step_sizes=args.step_sizes,
        n_leapfrog_steps=args.n_leapfrog,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 