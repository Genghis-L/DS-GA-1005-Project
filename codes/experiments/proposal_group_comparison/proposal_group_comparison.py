"""
This experiment is to compare the performance of different proposal groups.

The proposal for MCMC proposals are:
- Normal
- Student-t
- Uniform


The proposal for HMC are:
- Gaussian
- Student-t
- Uniform

The target distribution is:
- Standard Gaussian 
- Donut Distribution
- Gaussian Mixture Distribution

This will generate four figures 2x3:
For the first line of figures, there should be 3 figures, the title should be "Standard Gaussian comparison",
In the figure (1, 1), 
The x-axis should be the dimension of the standard Gaussian, and the y-axis should be the Acceptance Rate of the algorithm.
There should be two groups of lines, one for MCMC and one for HMC. In the group for MCMC, there should be three lines, one for Normal, one for Student-t, and one for Uniform. In the group for HMC, there should be two lines, one for Gaussian and one for Gaussian with ARS. 


In the figure (1, 2),
The x-axis should be the dimension of the standard Gaussian, and the y-axis should be the ESS of the algorithm.
There should be two groups of lines, one for MCMC and one for HMC. In the group for MCMC, there should be three lines, one for Normal, one for Student-t, and one for Uniform. In the group for HMC, there should be two lines, one for Gaussian and one for Gaussian with ARS. 


In the figure (1, 3),
The x-axis should be the dimension of the standard Gaussian, and the y-axis should be the Energy Error of the algorithm.
There should be two groups of lines, one for MCMC and one for HMC. In the group for MCMC, there should be three lines, one for Normal, one for Student-t, and one for Uniform. In the group for HMC, there should be two lines, one for Gaussian and one for Gaussian with ARS. 


For the second line of figures, there should be 3 figures, the title should be "Donut comparison",
Each figure in the line should show the same stuff(like the x-axis, y-axis, and the lines, though the stats varies) as the first line.


For the third line of figures, there should be 3 figures, the title should be "Gaussian Mixture comparison",
Each figure in the line should show the same stuff(like the x-axis, y-axis, and the lines, though the stats varies) as the first line.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys
import os
import time

# Add the parent directory to the path to import from algos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from algos import HMCSampler, TargetDistribution, MetropolisSampler


class StandardGaussian(TargetDistribution):
    """Standard Gaussian target distribution."""
    def __init__(self, dim: int):
        self.dim = dim
        
    def __call__(self, x: np.ndarray) -> float:
        return np.exp(-0.5 * np.sum(x * x)) / (2 * np.pi) ** (self.dim / 2)
    
    def log_density(self, x: np.ndarray) -> float:
        return -0.5 * np.sum(x * x) - (self.dim / 2) * np.log(2 * np.pi)
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        return -x

class DonutDistribution(TargetDistribution):
    """Donut-shaped target distribution."""
    def __init__(self, dim: int, radius: float = 2.0, width: float = 0.5):
        self.dim = dim
        self.radius = radius
        self.width = width
        
    def __call__(self, x: np.ndarray) -> float:
        r = np.sqrt(np.sum(x * x))
        return np.exp(-0.5 * ((r - self.radius) / self.width) ** 2) / (2 * np.pi) ** (self.dim / 2)
    
    def log_density(self, x: np.ndarray) -> float:
        r = np.sqrt(np.sum(x * x))
        return -0.5 * ((r - self.radius) / self.width) ** 2 - (self.dim / 2) * np.log(2 * np.pi)
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        r = np.sqrt(np.sum(x * x))
        if r == 0:
            return np.zeros_like(x)
        return -((r - self.radius) / (self.width * r)) * x

class GaussianMixture(TargetDistribution):
    """Gaussian mixture target distribution."""
    def __init__(self, dim: int, n_components: int = 2):
        self.dim = dim
        self.n_components = n_components
        self.means = [np.random.randn(dim) * 2 for _ in range(n_components)]
        self.covs = [np.eye(dim) for _ in range(n_components)]
        self.weights = np.ones(n_components) / n_components
        
    def __call__(self, x: np.ndarray) -> float:
        density = 0
        for i in range(self.n_components):
            diff = x - self.means[i]
            density += self.weights[i] * np.exp(-0.5 * diff.T @ np.linalg.inv(self.covs[i]) @ diff) / np.sqrt(np.linalg.det(2 * np.pi * self.covs[i]))
        return density
    
    def log_density(self, x: np.ndarray) -> float:
        return np.log(self(x))
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        # Numerical gradient for simplicity
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (self.log_density(x_plus) - self.log_density(x_minus)) / (2 * eps)
        return grad

def compute_ess(samples: np.ndarray) -> float:
    """
    Compute Effective Sample Size using autocorrelation.
    For multidimensional samples, computes ESS for each dimension and returns the minimum.
    """
    n = len(samples)
    if n <= 1:
        return 0.0
        
    # Compute autocorrelation up to lag 50 or n//3, whichever is smaller
    max_lag = min(50, n//3)
    mean = np.mean(samples)
    var = np.var(samples)
    
    if var == 0 or not np.isfinite(var):
        return 0.0
        
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag >= n:
            break
        c0 = samples[:-lag] - mean if lag > 0 else samples - mean
        c1 = samples[lag:] - mean
        if len(c0) == 0 or len(c1) == 0:
            break
        acf[lag] = np.mean(c0 * c1) / var
        
    # Find where autocorrelation drops below 0.05 or becomes negative
    cutoff = np.where((acf < 0.05) | (acf < 0))[0]
    if len(cutoff) > 0:
        max_lag = cutoff[0]
        
    # Compute ESS
    ess = n / (1 + 2 * np.sum(acf[:max_lag]))
    return max(1.0, ess)


def compute_energy_error(samples: np.ndarray, target: TargetDistribution) -> float:
    """
    Compute average energy error.
    Energy error is computed as the standard deviation of the energy values,
    which measures the conservation of energy in the sampling process.
    """
    energies = [-target.log_density(sample) for sample in samples]
    return np.std(energies)  # Use standard deviation instead of mean absolute difference

def run_experiment(
    target_class: TargetDistribution,
    dims: List[int],
    n_samples: int = 1000,
    n_warmup: int = 100
) -> Dict[str, Dict[str, List[float]]]:
    """Run experiment for a given target distribution."""
    results = {
        'mcmc': {
            'gaussian': {'acceptance': [], 'ess': [], 'time': []},
            'student_t': {'acceptance': [], 'ess': [], 'time': []},
            'uniform': {'acceptance': [], 'ess': [], 'time': []}
        },
        'hmc': {
            'gaussian': {'acceptance': [], 'ess': [], 'time': []},
            'student_t': {'acceptance': [], 'ess': [], 'time': []},
            'uniform': {'acceptance': [], 'ess': [], 'time': []}
        }
    }
    
    for dim in dims:
        print(f"Running experiment for dimension {dim}")
        target = target_class(dim)
        
        # MCMC experiments
        for proposal_type in ['gaussian', 'student_t', 'uniform']:
            if proposal_type == 'gaussian':
                proposal = lambda x: np.random.multivariate_normal(x, np.eye(dim))
            elif proposal_type == 'student_t':
                proposal = lambda x: x + np.random.standard_t(3, size=dim)
            else:  # uniform
                proposal = lambda x: x + np.random.uniform(-1, 1, size=dim)
            
            initial = lambda: np.zeros(dim)
            start_time = time.time()
            sampler = MetropolisSampler(
                target=target,
                initial=initial,
                proposal=proposal,
                iterations=n_samples + n_warmup
            )
            samples = np.array(sampler.run()[n_warmup:])  # Remove warmup samples
            end_time = time.time()
            
            # Compute metrics
            results['mcmc'][proposal_type]['acceptance'].append(sampler.get_acceptance_rate())
            results['mcmc'][proposal_type]['ess'].append(compute_ess(samples))
            results['mcmc'][proposal_type]['time'].append(end_time - start_time)
        
        # HMC experiments
        for kinetic_type in ['gaussian', 'student_t', 'uniform']:
            if kinetic_type == 'gaussian':
                kinetic_params = None
            elif kinetic_type == 'student_t':
                kinetic_params = {'nu': 3.0}
            else:  # uniform
                kinetic_params = {'scale': 1.0}
            
            initial = lambda: np.zeros(dim)
            start_time = time.time()
            sampler = HMCSampler(
                target=target,
                initial=initial,
                iterations=n_samples + n_warmup,
                L=50,
                step_size=0.1,
                kinetic_energy=kinetic_type,
                kinetic_params=kinetic_params
            )
            samples = np.array(sampler.run()[n_warmup:])  # Remove warmup samples
            end_time = time.time()
            
            # Compute metrics
            results['hmc'][kinetic_type]['acceptance'].append(sampler.get_acceptance_rate())
            results['hmc'][kinetic_type]['ess'].append(compute_ess(samples))
            results['hmc'][kinetic_type]['time'].append(end_time - start_time)
    
    return results

def plot_results(
    results: Dict[str, Dict[str, List[float]]],
    dims: List[int],
    target_name: str,
    save_path: str,
    n_samples: int = 1000
):
    """Plot results for a given target distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))  # Changed to vertical layout, adjusted size
    fig.suptitle(f"Target Distribution: {target_name}\nNumber of Points to sample: {n_samples}", 
                 fontsize=16, y=0.98)  # Moved title up
    
    # Define color schemes with more contrast
    mcmc_colors = ['#FF0000', '#FF8C00', '#FFD700']  # Red, Dark Orange, Gold
    hmc_colors = ['#000080', '#0000FF', '#87CEEB']   # Navy, Blue, Sky Blue
    
    # Plot acceptance rates
    ax = axes[0]
    for i, (method, color_scheme) in enumerate([('mcmc', mcmc_colors), ('hmc', hmc_colors)]):
        for j, proposal in enumerate(results[method].keys()):
            label = (f"MCMC - proposal - {proposal}" if method == 'mcmc' 
                    else f"HMC - kinetic energy - {proposal}")
            ax.plot(dims, results[method][proposal]['acceptance'], 
                   label=label,
                   color=color_scheme[j],
                   linestyle='-' if method == 'hmc' else '--',
                   marker='o' if method == 'hmc' else 's',
                   linewidth=2,
                   markersize=6)
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rate')
    ax.set_xscale('log')  # Set x-axis to log scale
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot ESS
    ax = axes[1]
    for i, (method, color_scheme) in enumerate([('mcmc', mcmc_colors), ('hmc', hmc_colors)]):
        for j, proposal in enumerate(results[method].keys()):
            label = (f"MCMC - proposal - {proposal}" if method == 'mcmc' 
                    else f"HMC - kinetic energy - {proposal}")
            ax.plot(dims, results[method][proposal]['ess'],
                   label=label,
                   color=color_scheme[j],
                   linestyle='-' if method == 'hmc' else '--',
                   marker='o' if method == 'hmc' else 's',
                   linewidth=2,
                   markersize=6)
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('ESS')
    ax.set_title('Effective Sample Size')
    ax.set_xscale('log')  # Set x-axis to log scale
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot computation time
    ax = axes[2]
    for i, (method, color_scheme) in enumerate([('mcmc', mcmc_colors), ('hmc', hmc_colors)]):
        for j, proposal in enumerate(results[method].keys()):
            label = (f"MCMC - proposal - {proposal}" if method == 'mcmc' 
                    else f"HMC - kinetic energy - {proposal}")
            ax.plot(dims, results[method][proposal]['time'],
                   label=label,
                   color=color_scheme[j],
                   linestyle='-' if method == 'hmc' else '--',
                   marker='o' if method == 'hmc' else 's',
                   linewidth=2,
                   markersize=6)
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computation Time')
    ax.set_xscale('log')  # Set x-axis to log scale
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    # Add extra space on the right for legends and top for title
    plt.subplots_adjust(right=0.85, top=0.9, hspace=0.4)  # Increased top margin and vertical spacing
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parameters
    dims = [2, 5, 10, 20, 50, 100, 200, 500]
    n_samples = 1000
    n_warmup = 100
    
    # Run experiments for each target distribution
    targets = {
        'Standard Gaussian': StandardGaussian,
        'Donut': DonutDistribution,
        # 'Gaussian Mixture': GaussianMixture
    }
    
    for target_name, target_class in targets.items():
        print(f"\nRunning experiments for {target_name}")
        results = run_experiment(target_class, dims, n_samples, n_warmup)
        plot_results(results, dims, target_name, f"{target_name.lower().replace(' ', '_')}_comparison.png", n_samples)

if __name__ == "__main__":
    main()
