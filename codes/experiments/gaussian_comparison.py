import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from tqdm import tqdm
from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import HMCSampler
from dists import NormalDistribution
import time

def effective_sample_size(samples: np.ndarray) -> float:
    """
    Compute effective sample size using autocorrelation.
    """
    n = len(samples)
    if n <= 1:
        return 0.0
        
    # Compute autocorrelation up to lag 50 or n//3, whichever is smaller
    max_lag = min(50, n // 3)
    mean = np.mean(samples)
    var = np.var(samples)
    
    if var == 0:
        return 0.0
        
    autocorr = np.correlate(samples - mean, samples - mean, mode='full')[n-1:n+max_lag]
    autocorr = autocorr / (n * var)
    
    # Find where autocorrelation drops below 0.05 or starts increasing
    cutoff = max_lag
    for i in range(1, max_lag):
        if autocorr[i] < 0.05 or autocorr[i] > autocorr[i-1]:
            cutoff = i
            break
            
    tau = 1 + 2 * np.sum(autocorr[1:cutoff])
    ess = n / tau
    return max(1, ess)  # Ensure ESS is at least 1

def run_dimension_experiment(
    dims: List[int],
    n_samples: int = 10000,
    n_runs: int = 5
) -> Dict:
    """
    Compare Metropolis and HMC performance across dimensions.
    
    Args:
        dims: List of dimensions to test
        n_samples: Number of samples per run
        n_runs: Number of runs per dimension (for variance estimation)
        
    Returns:
        Dictionary containing performance metrics
    """
    results = {
        'metropolis': {
            'ess': [], 'ess_std': [], 'time': [], 'accept_rate': []
        },
        'hmc': {
            'ess': [], 'ess_std': [], 'time': [], 'accept_rate': []
        }
    }
    
    for dim in tqdm(dims, desc="Testing dimensions"):
        # Setup target distribution (standard normal)
        target = NormalDistribution(
            mean=np.zeros(dim),
            cov=np.eye(dim)
        )
        initial = lambda: np.zeros(dim)
        
        # Storage for this dimension
        dim_results = {
            'metropolis': {'ess': [], 'time': [], 'accept': []},
            'hmc': {'ess': [], 'time': [], 'accept': []}
        }
        
        for _ in range(n_runs):
            # Run Metropolis
            start_time = time.time()
            metropolis = MetropolisSampler(
                target=target,
                initial=initial,
                proposal=NormalProposal(scale=2.4/np.sqrt(dim)),  # Optimal scaling
                iterations=n_samples
            )
            samples_m = np.array(metropolis.run())
            time_m = time.time() - start_time
            
            # Compute metrics for each dimension
            ess_m = np.mean([
                effective_sample_size(samples_m[:, i])
                for i in range(dim)
            ])
            accept_m = len(np.unique(samples_m, axis=0)) / len(samples_m)
            
            # Run HMC
            start_time = time.time()
            hmc = HMCSampler(
                target=target,
                initial=initial,
                iterations=n_samples,
                step_size=0.1/np.sqrt(dim),  # Scale with dimension
                L=int(np.pi/2 * np.sqrt(dim))  # Scale with dimension
            )
            samples_h = np.array(hmc.run())
            time_h = time.time() - start_time
            
            # Compute metrics
            ess_h = np.mean([
                effective_sample_size(samples_h[:, i])
                for i in range(dim)
            ])
            accept_h = len(np.unique(samples_h, axis=0)) / len(samples_h)
            
            # Store results
            dim_results['metropolis']['ess'].append(ess_m)
            dim_results['metropolis']['time'].append(time_m)
            dim_results['metropolis']['accept'].append(accept_m)
            
            dim_results['hmc']['ess'].append(ess_h)
            dim_results['hmc']['time'].append(time_h)
            dim_results['hmc']['accept'].append(accept_h)
        
        # Compute statistics across runs
        for sampler in ['metropolis', 'hmc']:
            results[sampler]['ess'].append(np.mean(dim_results[sampler]['ess']))
            results[sampler]['ess_std'].append(np.std(dim_results[sampler]['ess']))
            results[sampler]['time'].append(np.mean(dim_results[sampler]['time']))
            results[sampler]['accept_rate'].append(np.mean(dim_results[sampler]['accept']))
    
    return results

def plot_results(dims: List[int], results: Dict):
    """Plot comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Metropolis vs HMC Performance Comparison')
    
    # ESS plot
    ax = axes[0, 0]
    ax.errorbar(dims, results['metropolis']['ess'], 
                yerr=results['metropolis']['ess_std'],
                label='Metropolis', marker='o')
    ax.errorbar(dims, results['hmc']['ess'], 
                yerr=results['hmc']['ess_std'],
                label='HMC', marker='s')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Effective Sample Size')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    # ESS per second
    ax = axes[0, 1]
    ess_per_sec_m = np.array(results['metropolis']['ess']) / np.array(results['metropolis']['time'])
    ess_per_sec_h = np.array(results['hmc']['ess']) / np.array(results['hmc']['time'])
    ax.plot(dims, ess_per_sec_m, 'o-', label='Metropolis')
    ax.plot(dims, ess_per_sec_h, 's-', label='HMC')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('ESS per Second')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    # Acceptance rate
    ax = axes[1, 0]
    ax.plot(dims, results['metropolis']['accept_rate'], 'o-', label='Metropolis')
    ax.plot(dims, results['hmc']['accept_rate'], 's-', label='HMC')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Acceptance Rate')
    ax.legend()
    ax.grid(True)
    
    # Computation time
    ax = axes[1, 1]
    ax.plot(dims, results['metropolis']['time'], 'o-', label='Metropolis')
    ax.plot(dims, results['hmc']['time'], 's-', label='HMC')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('dimension_comparison.png')
    plt.close()

def main():
    # Test dimensions from 2 to 100
    dims = [2, 5, 10, 20, 50, 100]
    results = run_dimension_experiment(dims)
    plot_results(dims, results)
    
    # Print summary
    print("\nSummary of results:")
    print("\nEffective Sample Size (higher is better):")
    print("Dimension | Metropolis |    HMC")
    print("-" * 40)
    for i, d in enumerate(dims):
        print(f"{d:9d} | {results['metropolis']['ess'][i]:9.1f} | {results['hmc']['ess'][i]:9.1f}")
    
    print("\nAcceptance Rate:")
    print("Dimension | Metropolis |    HMC")
    print("-" * 40)
    for i, d in enumerate(dims):
        print(f"{d:9d} | {results['metropolis']['accept_rate'][i]:9.3f} | {results['hmc']['accept_rate'][i]:9.3f}")

if __name__ == "__main__":
    main() 