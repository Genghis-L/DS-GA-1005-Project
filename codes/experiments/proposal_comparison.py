import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.metropolis import MetropolisSampler, NormalProposal
from dists.normal import NormalDistribution

class StudentTProposal:
    """Student's t-distribution proposal with heavier tails."""
    def __init__(self, scale: float, df: float = 3.0):
        self.scale = scale
        self.df = df
        
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        jump = np.random.standard_t(df=self.df, size=sample.shape) * self.scale
        return sample + jump

class UniformProposal:
    """Uniform proposal distribution."""
    def __init__(self, scale: float):
        self.scale = scale
        
    def __call__(self, sample: np.ndarray) -> np.ndarray:
        jump = np.random.uniform(-self.scale, self.scale, size=sample.shape)
        return sample + jump

def compute_ess(samples: np.ndarray) -> float:
    """Compute Effective Sample Size using autocorrelation."""
    # Ensure samples is a 1D array
    samples = np.asarray(samples).flatten()
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
        # Ensure proper array shapes and handle edge cases
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
    return max(1.0, ess)  # Ensure ESS is at least 1

def plot_trajectories(
    samples_dict: dict,
    scale: float,
    output_prefix: str,
    dim: int,
    short_run: bool = True
):
    """Plot trajectory comparison between different proposal distributions."""
    n_proposals = len(samples_dict)
    fig, axes = plt.subplots(n_proposals, 1, figsize=(12, 4*n_proposals))
    if n_proposals == 1:
        axes = [axes]
    
    # For short run (first 200 iterations) or long run (first 1000 iterations)
    if short_run:
        n_samples = 200
        title = "First 200 iterations"
    else:
        n_samples = 1000
        title = f"First {n_samples} iterations"
    
    # Get global y-limits for consistent scale
    all_samples = np.concatenate([samples[:n_samples, 0] for samples in samples_dict.values()])
    ymin, ymax = all_samples.min(), all_samples.max()
    margin = 0.1 * (ymax - ymin)
    ymin -= margin
    ymax += margin
    
    # Plot each proposal's trajectory
    for ax, (prop_type, samples) in zip(axes, samples_dict.items()):
        traj = samples[:n_samples, 0]  # First coordinate
        ax.plot(range(n_samples), traj, 'k.', markersize=2)
        ax.set_title(f"{prop_type} Proposal (scale={scale})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("First position coordinate")
        ax.grid(True)
        ax.set_ylim(ymin, ymax)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    suffix = "short" if short_run else "long"
    plt.savefig(f"{output_prefix}_trajectory_{suffix}_scale{scale:.1f}_{dim}d.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_comparison(
    n_samples: int = 10000,
    dim: int = 1,
    target_std: float = 1.0,
    proposal_types: list = ['normal', 'student-t', 'uniform'],
    scales: list = [0.1, 0.5, 2.0],
    output_file: str = None
):
    """Run comparison of different proposal distributions."""
    # Setup target distribution (standard normal)
    target = NormalDistribution(
        mean=np.zeros(dim),
        cov=target_std**2 * np.eye(dim)
    )
    initial = lambda: np.zeros(dim)
    
    results = []
    
    for scale in scales:
        # Store samples for trajectory plots
        samples_dict = {}
        
        for prop_type in proposal_types:
            # Create proposal
            if prop_type == 'normal':
                proposal = NormalProposal(scale)
            elif prop_type == 'student-t':
                proposal = StudentTProposal(scale)
            else:  # uniform
                proposal = UniformProposal(scale)
                
            # Run sampler
            start_time = time.time()
            sampler = MetropolisSampler(
                target=target,
                initial=initial,
                proposal=proposal,
                iterations=n_samples
            )
            samples = np.array(sampler.run())
            end_time = time.time()
            
            # Store samples for trajectory plots
            samples_dict[prop_type] = samples
            
            # Compute metrics
            accept_rate = len(np.unique(samples, axis=0)) / len(samples)
            time_per_sample = (end_time - start_time) / n_samples
            
            # Compute ESS for each dimension and take the mean
            ess_values = []
            for i in range(dim):
                try:
                    ess_i = compute_ess(samples[:, i])
                    if np.isfinite(ess_i):
                        ess_values.append(ess_i)
                except:
                    continue
            
            ess = np.mean(ess_values) if ess_values else 0.0
            
            results.append({
                'proposal': prop_type,
                'scale': scale,
                'accept_rate': accept_rate,
                'time_per_sample': time_per_sample,
                'ess': ess,
                'ess_per_second': ess / (end_time - start_time) if ess > 0 else 0.0
            })
            
            print(f"\nResults for {prop_type.upper()} proposal (scale={scale}):")
            print(f"  Acceptance rate: {accept_rate:.2%}")
            print(f"  Time per sample: {time_per_sample*1e6:.2f} μs")
            print(f"  ESS: {ess:.1f}")
            print(f"  ESS/s: {ess/(end_time - start_time) if ess > 0 else 0.0:.1f}")
        
        # Generate trajectory plots for this scale
        plot_trajectories(samples_dict, scale, output_file.replace('.png', ''), dim, short_run=True)
        plot_trajectories(samples_dict, scale, output_file.replace('.png', ''), dim, short_run=False)
    
    # Plot comparison metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Proposal Distribution Comparison ({dim}D)")
    
    # Group results by proposal type
    for prop_type in proposal_types:
        prop_results = [r for r in results if r['proposal'] == prop_type]
        scales = [r['scale'] for r in prop_results]
        accept_rates = [r['accept_rate'] for r in prop_results]
        times = [r['time_per_sample'] for r in prop_results]
        ess_values = [r['ess'] for r in prop_results]
        ess_per_sec = [r['ess_per_second'] for r in prop_results]
        
        # Plot acceptance rate
        axes[0,0].plot(scales, accept_rates, 'o-', label=prop_type)
        axes[0,0].set_xlabel('Scale')
        axes[0,0].set_ylabel('Acceptance Rate')
        axes[0,0].set_title('Acceptance Rate vs Scale')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # Plot time per sample
        axes[0,1].plot(scales, np.array(times)*1e6, 'o-', label=prop_type)
        axes[0,1].set_xlabel('Scale')
        axes[0,1].set_ylabel('Time per Sample (μs)')
        axes[0,1].set_title('Computation Time vs Scale')
        axes[0,1].grid(True)
        axes[0,1].legend()
        
        # Plot ESS
        axes[1,0].plot(scales, ess_values, 'o-', label=prop_type)
        axes[1,0].set_xlabel('Scale')
        axes[1,0].set_ylabel('ESS')
        axes[1,0].set_title('Effective Sample Size vs Scale')
        axes[1,0].grid(True)
        axes[1,0].legend()
        
        # Plot ESS/s
        axes[1,1].plot(scales, ess_per_sec, 'o-', label=prop_type)
        axes[1,1].set_xlabel('Scale')
        axes[1,1].set_ylabel('ESS/s')
        axes[1,1].set_title('ESS per Second vs Scale')
        axes[1,1].grid(True)
        axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare different proposal distributions for Metropolis sampler")
    parser.add_argument("--dim", type=int, default=1,
                       help="Number of dimensions (default: 1)")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of samples to generate (default: 10000)")
    parser.add_argument("--target-std", type=float, default=1.0,
                       help="Standard deviation of target normal distribution (default: 1.0)")
    parser.add_argument("--proposal-types", type=str, nargs='+',
                       default=['normal', 'student-t', 'uniform'],
                       help="Types of proposal distributions to compare (default: normal student-t uniform)")
    parser.add_argument("--scales", type=float, nargs='+',
                       default=[0.1, 0.5, 1.0, 2.0, 5.0],
                       help="Scales to test for each proposal (default: 0.1 0.5 1.0 2.0 5.0)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file name (default: proposal_comparison_{dim}d.png)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'proposal_comparison_{args.dim}d.png'
    
    run_comparison(
        n_samples=args.n_samples,
        dim=args.dim,
        target_std=args.target_std,
        proposal_types=args.proposal_types,
        scales=args.scales,
        output_file=args.output
    )

if __name__ == "__main__":
    main() 