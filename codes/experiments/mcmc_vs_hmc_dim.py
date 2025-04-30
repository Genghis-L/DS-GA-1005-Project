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

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time
from scipy.stats import multivariate_normal, t, uniform

# Add the parent directory to the path to import from algos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos import HMCWithIntegrator, TargetDistribution, MetropolisSampler

class ProposalWrapper:
    """Wrapper for proposal distributions that counts FLOPs."""
    def __init__(self, proposal_type: str, dim: int):
        self.proposal_type = proposal_type
        self.dim = dim
        self.flop_count = 0
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.proposal_type == 'gaussian':
            # Random normal + addition for each dimension
            self.flop_count += 2 * self.dim
            return x + np.random.normal(size=self.dim)
        elif self.proposal_type == 'student_t':
            # t.rvs + addition for each dimension
            self.flop_count += 2 * self.dim
            return x + t.rvs(df=3, size=self.dim)
        else:  # uniform
            # uniform.rvs + addition for each dimension
            self.flop_count += 2 * self.dim
            return x + uniform.rvs(loc=-1, scale=2, size=self.dim)
    
    def get_flop_count(self) -> int:
        return self.flop_count

class StandardGaussian(TargetDistribution):
    """Standard Gaussian target distribution."""
    def __init__(self, dim: int):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.cov = np.eye(dim)
        self._mvn = multivariate_normal(mean=self.mean, cov=self.cov)
        self.flop_count = 0
        
    def __call__(self, x: np.ndarray) -> float:
        self.flop_count += self.dim * 2  # Square and sum operations
        return self._mvn.pdf(x)
    
    def log_density(self, x: np.ndarray) -> float:
        self.flop_count += self.dim * 2  # Square and sum operations
        return self._mvn.logpdf(x)
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        self.flop_count += self.dim  # Negation operation
        return -x
        
    def get_flop_count(self) -> int:
        return self.flop_count

class DonutDistribution(TargetDistribution):
    """Donut-shaped target distribution."""
    def __init__(self, dim: int, radius: float = 3.0, sigma2: float = 0.5):
        self.dim = dim
        self.radius = radius
        self.sigma2 = sigma2
        self.flop_count = 0
        
    def __call__(self, x: np.ndarray) -> float:
        # Count FLOPs for norm calculation (sum of squares + sqrt)
        self.flop_count += self.dim * 2 + 1
        r = np.sqrt(np.sum(x * x))
        # Count FLOPs for density calculation
        self.flop_count += 4  # subtraction, square, division, exp
        return np.exp(-(r - self.radius) ** 2 / self.sigma2)
    
    def log_density(self, x: np.ndarray) -> float:
        # Count FLOPs for norm calculation
        self.flop_count += self.dim * 2 + 1
        r = np.sqrt(np.sum(x * x))
        # Count FLOPs for log density
        self.flop_count += 3  # subtraction, square, division
        return -(r - self.radius) ** 2 / self.sigma2
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        # Count FLOPs for norm calculation
        self.flop_count += self.dim * 2 + 1
        r = np.sqrt(np.sum(x * x))
        if r == 0:
            return np.zeros_like(x)
        # Count FLOPs for gradient calculation
        self.flop_count += self.dim * 3 + 2  # division, subtraction, multiplication per dim
        return ((self.radius - r) / (self.sigma2 * r)) * x
        
    def get_flop_count(self) -> int:
        return self.flop_count

class GaussianMixture(TargetDistribution):
    """Gaussian mixture target distribution."""
    def __init__(self, dim: int, n_components: int = 2):
        self.dim = dim
        self.n_components = n_components
        self.means = [np.random.randn(dim) * 2 for _ in range(n_components)]
        self.covs = [np.eye(dim) for _ in range(n_components)]
        self.weights = np.ones(n_components) / n_components
        self.flop_count = 0
        
    def __call__(self, x: np.ndarray) -> float:
        self.flop_count += self.n_components * (self.dim * 3 + 2)  # diff, multiply, sum per component
        density = 0
        for i in range(self.n_components):
            diff = x - self.means[i]
            density += self.weights[i] * np.exp(-0.5 * diff.T @ np.linalg.inv(self.covs[i]) @ diff)
        return density
    
    def log_density(self, x: np.ndarray) -> float:
        self.flop_count += 1  # log operation
        return np.log(self(x))
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        # Numerical gradient uses 2*dim function evaluations
        self.flop_count += 2 * self.dim * (self.n_components * (self.dim * 3 + 2) + 1)
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (self.log_density(x_plus) - self.log_density(x_minus)) / (2 * eps)
        return grad
        
    def get_flop_count(self) -> int:
        return self.flop_count

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
            'gaussian': {'acceptance': [], 'ess': [], 'flops': []},
            'student_t': {'acceptance': [], 'ess': [], 'flops': []},
            'uniform': {'acceptance': [], 'ess': [], 'flops': []}
        },
        'hmc': {
            'gaussian': {'acceptance': [], 'ess': [], 'flops': []},
            'student_t': {'acceptance': [], 'ess': [], 'flops': []}
        }
    }
    
    for dim in dims:
        print(f"Running experiment for dimension {dim}")
        if target_class == DonutDistribution:
            target = target_class(dim, sigma2=0.5)
        else:
            target = target_class(dim)
        
        # MCMC experiments
        for proposal_type in ['gaussian', 'student_t', 'uniform']:
            # Create wrapped proposal
            proposal = ProposalWrapper(proposal_type, dim)
            
            # Set initial state based on target distribution
            if isinstance(target, DonutDistribution):
                initial = lambda: np.array([target.radius] + [0.0] * (dim-1))
            else:
                initial = lambda: np.zeros(dim)
            
            sampler = MetropolisSampler(
                target=target,
                initial=initial,
                proposal=proposal,
                iterations=n_samples + n_warmup
            )
            samples = np.array(sampler.run()[n_warmup:])
            
            # Total FLOPs = sampler FLOPs + proposal FLOPs + target FLOPs
            total_flops = sampler.get_flop_count() + proposal.get_flop_count() + target.get_flop_count()
            
            # Compute metrics
            results['mcmc'][proposal_type]['acceptance'].append(sampler.get_acceptance_rate())
            results['mcmc'][proposal_type]['ess'].append(compute_ess(samples))
            results['mcmc'][proposal_type]['flops'].append(total_flops)
        
        # HMC experiments
        for kinetic_type in ['gaussian', 'student_t']:
            if kinetic_type == 'gaussian':
                kinetic_params = None
                kinetic_name = 'gaussian'
            else:  # student_t
                kinetic_params = {'nu': 3.0}  # df = 3 for Student's t
                kinetic_name = 'student_t'
            
            # Set initial state based on target distribution
            if isinstance(target, DonutDistribution):
                initial = lambda: np.array([target.radius] + [0.0] * (dim-1))
            else:
                initial = lambda: np.zeros(dim)
            
            sampler = HMCWithIntegrator(
                target=target,
                initial=initial,
                iterations=n_samples + n_warmup,
                step_size=0.1,
                trajectory_length=0.5,  # s = ε * L
                kinetic_energy=kinetic_name,
                kinetic_params=kinetic_params
            )
            samples = np.array(sampler.run()[n_warmup:])  # Remove warmup samples
            
            # Compute metrics
            results['hmc'][kinetic_type]['acceptance'].append(sampler.get_acceptance_rate())
            results['hmc'][kinetic_type]['ess'].append(compute_ess(samples))
            results['hmc'][kinetic_type]['flops'].append(sampler.get_flop_count())
    
    return results

def plot_results(
    results: Dict[str, Dict[str, List[float]]],
    dims: List[int],
    target_name: str,
    save_path: str,
    n_samples: int = 1000
):
    """Plot results for a given target distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f"Target Distribution: {target_name}\nNumber of Points to sample: {n_samples}", 
                 fontsize=16, y=0.98)
    
    # Define color schemes with more contrast
    mcmc_colors = ['#FF0000', '#FF8C00', '#FFD700']  # Red, Dark Orange, Gold
    hmc_colors = ['#000080', '#0000FF']  # Navy, Blue
    
    # Plot acceptance rates
    ax = axes[0]
    # Plot MCMC results
    for j, proposal in enumerate(results['mcmc'].keys()):
        label = f"MCMC - {proposal}"
        ax.plot(dims, results['mcmc'][proposal]['acceptance'], 
               label=label,
               color=mcmc_colors[j],
               linestyle='--',
               marker='s',
               linewidth=2,
               markersize=6)
    
    # Plot HMC results with updated labels
    for j, (kinetic, color) in enumerate(zip(results['hmc'].keys(), hmc_colors)):
        if kinetic == 'gaussian':
            label = "HMC - Gaussian"
        elif kinetic == 'student_t':
            label = "HMC - Student-t (df=3)"
        
        ax.plot(dims, results['hmc'][kinetic]['acceptance'], 
               label=label,
               color=color,
               linestyle='-',
               marker='o',
               linewidth=2,
               markersize=6)
    
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rate')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot ESS (similar updates for other plots)
    ax = axes[1]
    for j, proposal in enumerate(results['mcmc'].keys()):
        label = f"MCMC - {proposal}"
        ax.plot(dims, results['mcmc'][proposal]['ess'],
               label=label,
               color=mcmc_colors[j],
               linestyle='--',
               marker='s',
               linewidth=2,
               markersize=6)
    
    for j, (kinetic, color) in enumerate(zip(results['hmc'].keys(), hmc_colors)):
        if kinetic == 'gaussian':
            label = "HMC - Gaussian"
        elif kinetic == 'student_t':
            label = "HMC - Student-t (df=3)"
        
        ax.plot(dims, results['hmc'][kinetic]['ess'],
               label=label,
               color=color,
               linestyle='-',
               marker='o',
               linewidth=2,
               markersize=6)
    
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('ESS')
    ax.set_title('Effective Sample Size')
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot FLOPs instead of computation time
    ax = axes[2]
    for j, proposal in enumerate(results['mcmc'].keys()):
        label = f"MCMC - {proposal}"
        ax.plot(dims, results['mcmc'][proposal]['flops'],
               label=label,
               color=mcmc_colors[j],
               linestyle='--',
               marker='s',
               linewidth=2,
               markersize=6)
    
    for j, (kinetic, color) in enumerate(zip(results['hmc'].keys(), hmc_colors)):
        if kinetic == 'gaussian':
            label = "HMC - Gaussian"
        elif kinetic == 'student_t':
            label = "HMC - Student-t (df=3)"
        
        ax.plot(dims, results['hmc'][kinetic]['flops'],
               label=label,
               color=color,
               linestyle='-',
               marker='o',
               linewidth=2,
               markersize=6)
    
    ax.set_xlabel('Dimension (log scale)')
    ax.set_ylabel('FLOPs')
    ax.set_title('Floating Point Operations')
    ax.set_xscale('log')
    ax.set_yscale('log')  # Use log scale for FLOPs
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.9, hspace=0.4)
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
        # 'Gaussian Mixture': GaussianMixture  # Commented out for now
    }
    
    for target_name, target_class in targets.items():
        print(f"\nRunning experiments for {target_name}")
        results = run_experiment(target_class, dims, n_samples, n_warmup)
        plot_results(results, dims, target_name, f"{target_name.lower().replace(' ', '_')}_comparison.png", n_samples)

if __name__ == "__main__":
    main()
