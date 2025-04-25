import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import HMCSampler
from dists.normal import NormalDistribution

def compute_ess(samples: np.ndarray) -> float:
    """Compute Effective Sample Size using autocorrelation."""
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

def plot_trajectories(
    metropolis_samples: np.ndarray,
    hmc_samples: np.ndarray,
    output_prefix: str,
    dim: int,
    short_run: bool = True
):
    """Plot trajectory comparison between Metropolis and HMC."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Get the first coordinate trajectory
    metro_traj = metropolis_samples[:, 0]
    hmc_traj = hmc_samples[:, 0]
    
    # For short run (first 200 iterations)
    if short_run:
        n_samples = 200
        title = "First 200 iterations"
    else:
        n_samples = 1000
        title = f"First {n_samples} iterations"
    
    # Plot Metropolis trajectory
    ax1.plot(range(n_samples), metro_traj[:n_samples], 'k.', markersize=2)
    ax1.set_title("Random-walk Metropolis")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("First position coordinate")
    ax1.grid(True)
    
    # Plot HMC trajectory
    ax2.plot(range(n_samples), hmc_traj[:n_samples], 'k.', markersize=2)
    ax2.set_title("Hamiltonian Monte Carlo")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("First position coordinate")
    ax2.grid(True)
    
    # Set y-limits to be the same for both plots
    ymin = min(metro_traj[:n_samples].min(), hmc_traj[:n_samples].min())
    ymax = max(metro_traj[:n_samples].max(), hmc_traj[:n_samples].max())
    margin = 0.1 * (ymax - ymin)
    ax1.set_ylim(ymin - margin, ymax + margin)
    ax2.set_ylim(ymin - margin, ymax + margin)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    suffix = "short" if short_run else "long"
    plt.savefig(f"{output_prefix}_trajectory_{suffix}_{dim}d.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_comparison(
    dimensions: List[int],
    n_samples: int,
    n_warmup: int,
    hmc_step_size: float,
    hmc_leapfrog_steps: int,
    metropolis_scale: float,
    output_prefix: str
) -> List[Dict[str, Any]]:
    """Run comparison between Metropolis and HMC samplers."""
    results = []
    
    for dim in dimensions:
        print(f"\nRunning comparison for {dim} dimensions...")
        
        # Setup target distribution (standard normal)
        target = NormalDistribution(mean=np.zeros(dim), cov=np.eye(dim))
        initial = lambda: np.zeros(dim)
        
        # Run Metropolis sampler
        start_time = time.time()
        metropolis = MetropolisSampler(
            target=target,
            initial=initial,
            proposal=NormalProposal(metropolis_scale),
            iterations=n_samples + n_warmup
        )
        metropolis_samples = np.array(metropolis.run())
        metro_time = time.time() - start_time
        
        # Run HMC sampler
        start_time = time.time()
        hmc = HMCSampler(
            target=target,
            initial=initial,
            step_size=hmc_step_size,
            L=hmc_leapfrog_steps,
            iterations=n_samples + n_warmup
        )
        hmc_samples = np.array(hmc.run())
        hmc_time = time.time() - start_time
        
        # Remove warmup samples
        metropolis_samples = metropolis_samples[n_warmup:]
        hmc_samples = hmc_samples[n_warmup:]
        
        # Generate trajectory plots for 2D and 100D cases
        if dim in [2, 100]:
            plot_trajectories(metropolis_samples, hmc_samples, output_prefix, dim, short_run=True)
            plot_trajectories(metropolis_samples, hmc_samples, output_prefix, dim, short_run=False)
        
        # Compute metrics
        result = {
            'dimension': dim,
            'metropolis': {
                'ess': np.mean([compute_ess(metropolis_samples[:, i]) for i in range(dim)]),
                'time': metro_time,
                'accept_rate': len(np.unique(metropolis_samples, axis=0)) / len(metropolis_samples)
            },
            'hmc': {
                'ess': np.mean([compute_ess(hmc_samples[:, i]) for i in range(dim)]),
                'time': hmc_time,
                'accept_rate': len(np.unique(hmc_samples, axis=0)) / len(hmc_samples)
            }
        }
        
        results.append(result)
        
        # Print results
        print(f"\nResults for {dim} dimensions:")
        print("Metropolis:")
        print(f"  ESS: {result['metropolis']['ess']:.1f}")
        print(f"  Time: {result['metropolis']['time']:.2f}s")
        print(f"  Accept Rate: {result['metropolis']['accept_rate']:.2%}")
        print("HMC:")
        print(f"  ESS: {result['hmc']['ess']:.1f}")
        print(f"  Time: {result['hmc']['time']:.2f}s")
        print(f"  Accept Rate: {result['hmc']['accept_rate']:.2%}")
    
    return results

def plot_comparison_metrics(results: List[Dict[str, Any]], output_prefix: str):
    """Plot comparison metrics across dimensions."""
    dimensions = [r['dimension'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Metropolis vs HMC Comparison")
    
    # ESS
    axes[0,0].plot(dimensions, [r['metropolis']['ess'] for r in results], 'o-', label='Metropolis')
    axes[0,0].plot(dimensions, [r['hmc']['ess'] for r in results], 's-', label='HMC')
    axes[0,0].set_xlabel('Dimension')
    axes[0,0].set_ylabel('ESS')
    axes[0,0].set_title('Effective Sample Size vs Dimension')
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True)
    axes[0,0].legend()
    
    # Time
    axes[0,1].plot(dimensions, [r['metropolis']['time'] for r in results], 'o-', label='Metropolis')
    axes[0,1].plot(dimensions, [r['hmc']['time'] for r in results], 's-', label='HMC')
    axes[0,1].set_xlabel('Dimension')
    axes[0,1].set_ylabel('Time (s)')
    axes[0,1].set_title('Computation Time vs Dimension')
    axes[0,1].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True)
    axes[0,1].legend()
    
    # Acceptance Rate
    axes[1,0].plot(dimensions, [r['metropolis']['accept_rate'] for r in results], 'o-', label='Metropolis')
    axes[1,0].plot(dimensions, [r['hmc']['accept_rate'] for r in results], 's-', label='HMC')
    axes[1,0].set_xlabel('Dimension')
    axes[1,0].set_ylabel('Acceptance Rate')
    axes[1,0].set_title('Acceptance Rate vs Dimension')
    axes[1,0].set_xscale('log')
    axes[1,0].grid(True)
    axes[1,0].legend()
    
    # ESS per second
    axes[1,1].plot(dimensions, 
                   [r['metropolis']['ess']/r['metropolis']['time'] for r in results], 
                   'o-', label='Metropolis')
    axes[1,1].plot(dimensions, 
                   [r['hmc']['ess']/r['hmc']['time'] for r in results], 
                   's-', label='HMC')
    axes[1,1].set_xlabel('Dimension')
    axes[1,1].set_ylabel('ESS/s')
    axes[1,1].set_title('ESS per Second vs Dimension')
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare Metropolis and HMC samplers")
    parser.add_argument("--dimensions", type=int, nargs='+',
                       default=[2, 5, 10, 20, 50, 100],
                       help="Dimensions to test (default: 2 5 10 20 50 100)")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples per dimension (default: 1000)")
    parser.add_argument("--n-warmup", type=int, default=1000,
                       help="Number of warmup samples (default: 1000)")
    parser.add_argument("--hmc-step-size", type=float, default=0.1,
                       help="Step size for HMC (default: 0.1)")
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=50,
                       help="Number of leapfrog steps for HMC (default: 50)")
    parser.add_argument("--metropolis-scale", type=float, default=0.1,
                       help="Scale for Metropolis proposal (default: 0.1)")
    parser.add_argument("--output", type=str, default="gaussian_comparison",
                       help="Output file prefix (default: gaussian_comparison)")
    
    args = parser.parse_args()
    
    results = run_comparison(
        dimensions=args.dimensions,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        hmc_step_size=args.hmc_step_size,
        hmc_leapfrog_steps=args.hmc_leapfrog_steps,
        metropolis_scale=args.metropolis_scale,
        output_prefix=args.output
    )
    
    plot_comparison_metrics(results, args.output)

if __name__ == "__main__":
    main() 