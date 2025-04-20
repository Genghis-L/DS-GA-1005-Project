import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import seaborn as sns

def plot_samples_1d(
    samples: List[np.ndarray],
    target: Optional[callable] = None,
    x_range: tuple = (-5, 5),
    title: str = "MCMC Samples",
    save_path: Optional[str] = None
):
    """Plot 1D samples with optional target density overlay."""
    plt.figure(figsize=(10, 6))
    
    # Plot samples
    sns.histplot(
        [s[0] for s in samples],
        stat="density",
        bins=50,
        alpha=0.5,
        label="Samples"
    )
    
    # Plot target density if provided
    if target is not None:
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = [target(np.array([xi])) for xi in x]
        plt.plot(x, y, 'r-', label="Target Density")
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_samples_2d(
    samples: List[np.ndarray],
    target: Optional[callable] = None,
    x_range: tuple = (-5, 5),
    y_range: tuple = (-5, 5),
    title: str = "MCMC Samples",
    save_path: Optional[str] = None
):
    """Plot 2D samples with optional target density contour."""
    plt.figure(figsize=(10, 8))
    
    # Plot samples
    x = [s[0] for s in samples]
    y = [s[1] for s in samples]
    plt.scatter(x, y, alpha=0.5, s=1, label="Samples")
    
    # Plot target density contour if provided
    if target is not None:
        xx, yy = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 100),
            np.linspace(y_range[0], y_range[1], 100)
        )
        z = np.array([[target(np.array([xi, yi])) for xi in xx[0]] for yi in yy[:, 0]])
        plt.contour(xx, yy, z, levels=10, colors='r', alpha=0.5)
    
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_trace(samples: List[np.ndarray], title: str = "Trace Plot", save_path: Optional[str] = None):
    """Plot trace of samples over iterations."""
    plt.figure(figsize=(12, 6))
    
    samples_array = np.array(samples)
    for i in range(samples_array.shape[1]):
        plt.plot(samples_array[:, i], alpha=0.7, label=f"Dimension {i+1}")
    
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show() 