import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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

class MCMCVisualizer:
    """Real-time visualizer for MCMC sampling."""
    
    def __init__(
        self,
        target: Optional[callable] = None,
        x_range: tuple = (-5, 5),
        y_range: tuple = (-5, 5),
        update_interval: int = 100,
        max_samples: int = 10000
    ):
        """
        Initialize the visualizer.
        
        Args:
            target: Target distribution callable
            x_range: Range for x-axis
            y_range: Range for y-axis
            update_interval: Number of samples between plot updates
            max_samples: Maximum number of samples to show
        """
        self.target = target
        self.x_range = x_range
        self.y_range = y_range
        self.update_interval = update_interval
        self.max_samples = max_samples
        
        # Initialize plot
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(15, 5))
        
        # Create subplots
        self.ax_samples = self.fig.add_subplot(121)  # For samples
        self.ax_trace = self.fig.add_subplot(122)    # For trace
        
        # Initialize data storage
        self.samples = []
        self.sample_plot = None
        self.trace_lines = []
        self.trace_data = []
        
        # Setup plots
        self._setup_plots()
        
    def _setup_plots(self):
        """Setup initial plot layout."""
        # Setup sample plot
        self.ax_samples.grid(True)
        self.ax_samples.set_title("MCMC Samples")
        
        # Setup trace plot
        self.ax_trace.grid(True)
        self.ax_trace.set_title("Trace Plot")
        self.ax_trace.set_xlabel("Iteration")
        self.ax_trace.set_ylabel("Value")
        
        if self.target is not None:
            # Plot target density for 1D
            if len(self.samples) > 0 and len(self.samples[0]) == 1:
                x = np.linspace(self.x_range[0], self.x_range[1], 1000)
                y = [self.target(np.array([xi])) for xi in x]
                self.ax_samples.plot(x, y, 'r-', label="Target Density")
            # Plot contours for 2D
            elif len(self.samples) > 0 and len(self.samples[0]) == 2:
                xx, yy = np.meshgrid(
                    np.linspace(self.x_range[0], self.x_range[1], 100),
                    np.linspace(self.y_range[0], self.y_range[1], 100)
                )
                z = np.array([[self.target(np.array([xi, yi])) 
                             for xi in xx[0]] for yi in yy[:, 0]])
                self.ax_samples.contour(xx, yy, z, levels=10, colors='r', alpha=0.5)
        
        plt.tight_layout()
        
    def update(self, sample: np.ndarray):
        """Update plots with new sample."""
        self.samples.append(sample)
        self.trace_data.append(sample)
        
        # Only update every update_interval samples
        if len(self.samples) % self.update_interval == 0:
            self._update_plots()
            
    def _update_plots(self):
        """Update both sample and trace plots."""
        # Clear previous plots
        self.ax_samples.clear()
        self.ax_trace.clear()
        
        # Limit number of samples shown
        show_samples = self.samples[-self.max_samples:]
        
        # Update sample plot
        if len(self.samples[0]) == 1:
            # 1D case: histogram
            self.ax_samples.hist(
                [s[0] for s in show_samples],
                bins=50,
                density=True,
                alpha=0.5,
                label="Samples"
            )
            if self.target is not None:
                x = np.linspace(self.x_range[0], self.x_range[1], 1000)
                y = [self.target(np.array([xi])) for xi in x]
                self.ax_samples.plot(x, y, 'r-', label="Target Density")
        else:
            # 2D case: scatter plot
            x = [s[0] for s in show_samples]
            y = [s[1] for s in show_samples]
            self.ax_samples.scatter(x, y, alpha=0.5, s=1)
            if self.target is not None:
                xx, yy = np.meshgrid(
                    np.linspace(self.x_range[0], self.x_range[1], 100),
                    np.linspace(self.y_range[0], self.y_range[1], 100)
                )
                z = np.array([[self.target(np.array([xi, yi])) 
                             for xi in xx[0]] for yi in yy[:, 0]])
                self.ax_samples.contour(xx, yy, z, levels=10, colors='r', alpha=0.5)
        
        # Update trace plot
        samples_array = np.array(self.trace_data)
        for i in range(samples_array.shape[1]):
            self.ax_trace.plot(
                samples_array[:, i],
                alpha=0.7,
                label=f"Dimension {i+1}"
            )
        
        # Update labels and legends
        self.ax_samples.set_title(f"MCMC Samples (n={len(self.samples)})")
        self.ax_samples.grid(True)
        self.ax_samples.legend()
        
        self.ax_trace.set_title("Trace Plot")
        self.ax_trace.set_xlabel("Iteration")
        self.ax_trace.set_ylabel("Value")
        self.ax_trace.grid(True)
        self.ax_trace.legend()
        
        plt.tight_layout()
        plt.pause(0.01)  # Small pause to allow plot to update
        
    def finalize(self, save_path: Optional[str] = None):
        """Finalize the plot and optionally save it."""
        if save_path:
            plt.savefig(save_path)
        plt.show() 