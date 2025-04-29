"""
This experiment is to compare the performance of different step sizes for both MCMC and HMC.

For MCMC:
- Gaussian proposal with different scales (0.1, 0.5, 1.0, 2.0, 5.0)
- Student-t proposal with different scales (0.1, 0.5, 1.0, 2.0, 5.0)
- Uniform proposal with different ranges ([-0.1,0.1], [-0.5,0.5], [-1,1], [-2,2], [-5,5])

For HMC:
- Gaussian kinetic energy with different step sizes (0.01, 0.05, 0.1, 0.2, 0.5)
- Student-t kinetic energy with different step sizes (0.01, 0.05, 0.1, 0.2, 0.5)
- Uniform kinetic energy with different step sizes (0.01, 0.05, 0.1, 0.2, 0.5)

The target distribution is Standard Gaussian in different dimensions.
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
from .proposal_group_comparison import StandardGaussian, compute_ess

def run_experiment(
    dims: List[int],
    n_samples: int = 1000,
    n_warmup: int = 100
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Run experiment for different step sizes."""
    # Define step sizes/scales for each method
    mcmc_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    hmc_step_sizes = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    results = {
        'mcmc': {
            'gaussian': {str(scale): {'acceptance': [], 'ess': [], 'time': []} for scale in mcmc_scales},
            'student_t': {str(scale): {'acceptance': [], 'ess': [], 'time': []} for scale in mcmc_scales},
            'uniform': {str(scale): {'acceptance': [], 'ess': [], 'time': []} for scale in mcmc_scales}
        },
        'hmc': {
            'gaussian': {str(step): {'acceptance': [], 'ess': [], 'time': []} for step in hmc_step_sizes},
            'student_t': {str(step): {'acceptance': [], 'ess': [], 'time': []} for step in hmc_step_sizes},
            'uniform': {str(step): {'acceptance': [], 'ess': [], 'time': []} for step in hmc_step_sizes}
        }
    }
    
    for dim in dims:
        print(f"Running experiment for dimension {dim}")
        target = StandardGaussian(dim)
        
        # MCMC experiments
        for proposal_type in ['gaussian', 'student_t', 'uniform']:
            for scale in mcmc_scales:
                if proposal_type == 'gaussian':
                    proposal = lambda x: np.random.multivariate_normal(x, (scale**2) * np.eye(dim))
                elif proposal_type == 'student_t':
                    proposal = lambda x: x + scale * np.random.standard_t(3, size=dim)
                else:  # uniform
                    proposal = lambda x: x + np.random.uniform(-scale, scale, size=dim)
                
                initial = lambda: np.zeros(dim)
                start_time = time.time()
                sampler = MetropolisSampler(
                    target=target,
                    initial=initial,
                    proposal=proposal,
                    iterations=n_samples + n_warmup
                )
                samples = np.array(sampler.run()[n_warmup:])
                end_time = time.time()
                
                # Store results
                results['mcmc'][proposal_type][str(scale)]['acceptance'].append(sampler.get_acceptance_rate())
                results['mcmc'][proposal_type][str(scale)]['ess'].append(compute_ess(samples))
                results['mcmc'][proposal_type][str(scale)]['time'].append(end_time - start_time)
        
        # HMC experiments
        for kinetic_type in ['gaussian', 'student_t', 'uniform']:
            for step_size in hmc_step_sizes:
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
                    L=50,  # Fixed number of leapfrog steps
                    step_size=step_size,
                    kinetic_energy=kinetic_type,
                    kinetic_params=kinetic_params
                )
                samples = np.array(sampler.run()[n_warmup:])
                end_time = time.time()
                
                # Store results
                results['hmc'][kinetic_type][str(step_size)]['acceptance'].append(sampler.get_acceptance_rate())
                results['hmc'][kinetic_type][str(step_size)]['ess'].append(compute_ess(samples))
                results['hmc'][kinetic_type][str(step_size)]['time'].append(end_time - start_time)
    
    return results

def plot_results(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    dims: List[int],
    save_path: str,
    n_samples: int = 1000
):
    """Plot results comparing different step sizes."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"Step Size Comparison\nNumber of Points to sample: {n_samples}", fontsize=16, y=0.98)
    
    # Define color schemes - use color gradients for different step sizes
    mcmc_colors = plt.cm.Reds(np.linspace(0.3, 0.9, 5))  # 5 different scales
    hmc_colors = plt.cm.Blues(np.linspace(0.3, 0.9, 5))  # 5 different step sizes
    
    # Plot for each proposal/kinetic type
    for i, dist_type in enumerate(['gaussian', 'student_t', 'uniform']):
        # MCMC plots (left column)
        ax = axes[i, 0]
        for j, (scale, metrics) in enumerate(results['mcmc'][dist_type].items()):
            ax.plot(dims, metrics['acceptance'], 
                   label=f"Scale = {scale}",
                   color=mcmc_colors[j],
                   linestyle='--',
                   marker='s',
                   linewidth=2,
                   markersize=6)
        ax.set_xlabel('Dimension (log scale)')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title(f'MCMC - {dist_type} proposal')
        ax.set_xscale('log')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # HMC plots (right column)
        ax = axes[i, 1]
        for j, (step, metrics) in enumerate(results['hmc'][dist_type].items()):
            ax.plot(dims, metrics['acceptance'], 
                   label=f"Step size = {step}",
                   color=hmc_colors[j],
                   linestyle='-',
                   marker='o',
                   linewidth=2,
                   markersize=6)
        ax.set_xlabel('Dimension (log scale)')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title(f'HMC - {dist_type} kinetic energy')
        ax.set_xscale('log')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parameters
    dims = [2, 5, 10, 20, 50, 100]
    n_samples = 1000
    n_warmup = 100
    
    print("\nRunning step size comparison experiments")
    results = run_experiment(dims, n_samples, n_warmup)
    plot_results(results, dims, "step_size_comparison.png", n_samples)

if __name__ == "__main__":
    main()