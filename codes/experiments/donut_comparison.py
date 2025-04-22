import os
import sys
# Add parent directory to path so we can import algos, basically it appends the /codes directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from algos.metropolis import MetropolisSampler, NormalProposal
from algos.hmc import HMCSampler, TargetDistribution

class DonutDistribution(TargetDistribution):
    """Donut-shaped distribution (annular Gaussian)."""
    
    def __init__(self, radius: float = 3.0, sigma2: float = 0.05):
        self.radius = radius
        self.sigma2 = sigma2
        
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
    x = np.linspace(extent[0], extent[1], n_points)
    y = np.linspace(extent[2], extent[3], n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            Z[i,j] = target(np.array([X[i,j], Y[i,j]]))
    
    levels = np.linspace(0, Z.max(), 10)
    ax.contour(X, Y, Z, levels=levels, colors='pink', alpha=0.5)

def run_and_plot_comparison(n_samples: int = 166):
    """Run and plot comparison of RW-Metropolis with different scales and HMC."""
    # Setup target and initial point
    target = DonutDistribution(radius=3.0, sigma2=0.05)
    initial = lambda: np.array([3.0, 0.0])
    
    # Run samplers
    # RW-Metropolis with small scale
    metropolis_small = MetropolisSampler(
        target=target,
        initial=initial,
        proposal=NormalProposal(scale=0.05),
        iterations=n_samples
    )
    samples_small = np.array(metropolis_small.run())
    accept_rate_small = len(np.unique(samples_small, axis=0)) / len(samples_small)
    
    # RW-Metropolis with large scale
    metropolis_large = MetropolisSampler(
        target=target,
        initial=initial,
        proposal=NormalProposal(scale=1.0),
        iterations=n_samples
    )
    samples_large = np.array(metropolis_large.run())
    accept_rate_large = len(np.unique(samples_large, axis=0)) / len(samples_large)
    
    # HMC
    hmc = HMCSampler(
        target=target,
        initial=initial,
        iterations=n_samples,
        step_size=0.1,
        L=50
    )
    samples_hmc = np.array(hmc.run())
    accept_rate_hmc = len(np.unique(samples_hmc, axis=0)) / len(samples_hmc)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot settings
    extent = (-4, 4, -4, 4)
    titles = [
        f"RW-Metropolis - scale = 0.05\n# samples: {n_samples}\nAccept rate: {accept_rate_small:.2%}",
        f"RW-Metropolis - scale = 1.0\n# samples: {n_samples}\nAccept rate: {accept_rate_large:.2%}",
        f"HMC\n# samples: {n_samples}\nAccept rate: {accept_rate_hmc:.2%}"
    ]
    
    for ax, samples, title in zip(axes, [samples_small, samples_large, samples_hmc], titles):
        # Plot density contours
        plot_density_contours(ax, target, extent)
        
        # Plot samples
        ax.scatter(samples[:, 0], samples[:, 1], c='blue', alpha=0.5, s=20)
        
        # Settings
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('donut_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_and_plot_comparison() 