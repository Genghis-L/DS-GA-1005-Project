import os
import sys
import argparse
# Add parent directory to path so we can import algos, basically it appends the /codes directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import HMCSampler, TargetDistribution

class DonutDistribution(TargetDistribution):
    """N-dimensional shell distribution (generalized donut)."""
    
    def __init__(self, radius: float = 3.0, sigma2: float = 0.05, dim: int = 2):
        """
        Initialize N-dimensional shell distribution.
        
        Args:
            radius: Target radius of the shell
            sigma2: Variance parameter controlling shell thickness
            dim: Number of dimensions
        """
        self.radius = radius
        self.sigma2 = sigma2
        self.dim = dim
        
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate unnormalized density."""
        r = np.linalg.norm(x)
        return np.exp(-(r - self.radius) ** 2 / self.sigma2)
    
    def log_density(self, x: np.ndarray) -> float:
        """Log of unnormalized density."""
        r = np.linalg.norm(x)
        return -(r - self.radius) ** 2 / self.sigma2
    
    def grad_log_density(self, x: np.ndarray) -> np.ndarray:
        """Gradient of log density."""
        r = np.linalg.norm(x)
        if r == 0:
            return np.zeros_like(x)
        return 2 * x * (self.radius / r - 1) / self.sigma2

def plot_density_contours(ax, target, extent=(-4, 4, -4, 4), n_points=100):
    """Plot contours of the target density."""
    if target.dim != 2:
        raise ValueError("Contour plotting only supported for 2D distributions")
        
    x = np.linspace(extent[0], extent[1], n_points)
    y = np.linspace(extent[2], extent[3], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            Z[i,j] = target(np.array([X[i,j], Y[i,j]]))
    
    levels = np.linspace(0, Z.max(), 10)
    ax.contour(X, Y, Z, levels=levels, colors='pink', alpha=0.5)

def plot_samples_nd(samples: np.ndarray, target_dim: int, save_path: str = None):
    """Plot samples for different dimensions."""
    if target_dim == 2:
        # For 2D, plot as before
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        return axes
    else:
        # For higher dimensions, plot pairwise projections of first 3 dimensions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{target_dim}D Shell Distribution - First 3 Dimensions Projections")
        
        # Define which dimensions to plot
        dims = [(0,1), (1,2), (0,2)]
        for ax, (i,j) in zip(axes, dims):
            ax.scatter(samples[:, i], samples[:, j], c='blue', alpha=0.5, s=20)
            ax.set_xlabel(f'Dimension {i+1}')
            ax.set_ylabel(f'Dimension {j+1}')
            # Draw target radius circle
            circle = plt.Circle((0, 0), 3.0, fill=False, color='pink', alpha=0.5)
            ax.add_artist(circle)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.grid(True)
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return axes

def main():
    parser = argparse.ArgumentParser(description="Compare MCMC samplers on n-dimensional donut distribution")
    parser.add_argument("--dim", type=int, default=2,
                       help="Number of dimensions (default: 2)")
    parser.add_argument("--n-samples", type=int, default=166,
                       help="Number of samples to generate (default: 166)")
    parser.add_argument("--radius", type=float, default=3.0,
                       help="Target radius of the shell (default: 3.0)")
    parser.add_argument("--sigma2", type=float, default=0.05,
                       help="Variance parameter controlling shell thickness (default: 0.05)")
    parser.add_argument("--metropolis-scales", type=float, nargs=2, default=[0.05, 1.0],
                       help="Two scale values for Metropolis sampler (default: 0.05 1.0)")
    parser.add_argument("--hmc-step-size", type=float, default=0.1,
                       help="Step size for HMC (default: 0.1)")
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=50,
                       help="Number of leapfrog steps for HMC (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file name (default: donut_comparison_{dim}d.png)")
    
    args = parser.parse_args()
    
    if args.dim < 2:
        parser.error("Dimension must be at least 2")
    
    if args.output is None:
        args.output = f'donut_comparison_{args.dim}d.png'
    
    print(f"Running comparison for {args.dim} dimensions...")
    print(f"Parameters:")
    print(f"  - Number of samples: {args.n_samples}")
    print(f"  - Target radius: {args.radius}")
    print(f"  - Shell thickness (sigma²): {args.sigma2}")
    print(f"  - Metropolis scales: {args.metropolis_scales}")
    print(f"  - HMC step size: {args.hmc_step_size}")
    print(f"  - HMC leapfrog steps: {args.hmc_leapfrog_steps}")
    print(f"  - Output file: {args.output}")
    
    run_and_plot_comparison(
        n_samples=args.n_samples,
        dim=args.dim,
        radius=args.radius,
        sigma2=args.sigma2,
        metropolis_scales=args.metropolis_scales,
        hmc_step_size=args.hmc_step_size,
        hmc_leapfrog_steps=args.hmc_leapfrog_steps,
        output_file=args.output
    )

def run_and_plot_comparison(
    n_samples: int = 166,
    dim: int = 2,
    radius: float = 3.0,
    sigma2: float = 0.05,
    metropolis_scales: list = [0.05, 1.0],
    hmc_step_size: float = 0.1,
    hmc_leapfrog_steps: int = 50,
    output_file: str = None
):
    """Run and plot comparison of RW-Metropolis with different scales and HMC."""
    # Setup target and initial point
    target = DonutDistribution(radius=radius, sigma2=sigma2, dim=dim)
    initial = lambda: np.array([radius] + [0.0] * (dim-1))  # Start on the x-axis
    
    # Run samplers
    # RW-Metropolis with small scale
    metropolis_small = MetropolisSampler(
        target=target,
        initial=initial,
        proposal=NormalProposal(scale=metropolis_scales[0]),
        iterations=n_samples
    )
    samples_small = np.array(metropolis_small.run())
    accept_rate_small = len(np.unique(samples_small, axis=0)) / len(samples_small)
    
    # RW-Metropolis with large scale
    metropolis_large = MetropolisSampler(
        target=target,
        initial=initial,
        proposal=NormalProposal(scale=metropolis_scales[1]),
        iterations=n_samples
    )
    samples_large = np.array(metropolis_large.run())
    accept_rate_large = len(np.unique(samples_large, axis=0)) / len(samples_large)
    
    # HMC
    hmc = HMCSampler(
        target=target,
        initial=initial,
        iterations=n_samples,
        step_size=hmc_step_size,
        L=hmc_leapfrog_steps
    )
    samples_hmc = np.array(hmc.run())
    accept_rate_hmc = len(np.unique(samples_hmc, axis=0)) / len(samples_hmc)
    
    # Create plots
    axes = plot_samples_nd(samples_hmc, dim)  # Get axes for appropriate dimension
    
    # Plot settings
    titles = [
        f"RW-Metropolis - scale = {metropolis_scales[0]}\n# samples: {n_samples}\nAccept rate: {accept_rate_small:.2%}",
        f"RW-Metropolis - scale = {metropolis_scales[1]}\n# samples: {n_samples}\nAccept rate: {accept_rate_large:.2%}",
        f"HMC\n# samples: {n_samples}\nAccept rate: {accept_rate_hmc:.2%}"
    ]
    
    for ax, samples, title in zip(axes, [samples_small, samples_large, samples_hmc], titles):
        if dim == 2:
            # For 2D, plot contours
            plot_density_contours(ax, target)
            ax.scatter(samples[:, 0], samples[:, 1], c='blue', alpha=0.5, s=20)
        ax.set_title(title)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 