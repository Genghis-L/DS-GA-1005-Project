# Experiment Settings for Step Size Comparison Study

## General Settings
| Parameter | Value |
|-----------|-------|
| Number of Points to Sample | 1000 |
| Warmup Samples | 100 |
| Dimensions Tested | [2, 5, 10, 20, 50, 100] |

## Target Distribution

### Standard Gaussian
| Parameter | Value |
|-----------|-------|
| Type | Multivariate Normal |
| Mean | Zero vector |
| Covariance | Identity matrix |

## MCMC Settings

### Proposal Distributions and Scales
| Proposal Type | Scale Values | Implementation Details |
|--------------|--------------|----------------------|
| Gaussian | [0.1, 0.5, 1.0, 2.0, 5.0] | \(\mathcal{N}(x, \sigma^2 I)\) where \(\sigma\) is the scale |
| Student-t | [0.1, 0.5, 1.0, 2.0, 5.0] | \(x + \text{scale} \cdot t_3\) where \(t_3\) has 3 degrees of freedom |
| Uniform | [0.1, 0.5, 1.0, 2.0, 5.0] | \(x + U(-\text{scale}, \text{scale})\) |

## HMC Settings

### Leapfrog Integrator Parameters
| Parameter | Value |
|-----------|-------|
| Number of Steps (L) | 50 (fixed) |
| Step Sizes (ε) | [0.01, 0.05, 0.1, 0.2, 0.5] |

### Kinetic Energy Distributions
| Type | Parameters | Implementation Details |
|------|------------|----------------------|
| Gaussian | None | Standard normal momentum |
| Student-t | ν = 3.0 | Student-t with 3 degrees of freedom |
| Uniform | scale = 1.0 | Uniform distribution on [-1, 1] |

## Performance Metrics
- **Acceptance Rate**: Proportion of proposed moves that are accepted
- **Effective Sample Size (ESS)**: Measure of the effective number of independent samples
- **Computation Time**: Wall clock time for sampling

## Visualization Settings
- **Plot Layout**: 3×2 grid
  - Left column: MCMC results for each proposal type
  - Right column: HMC results for each kinetic energy type
- **Color Schemes**:
  - MCMC: Red color gradient
  - HMC: Blue color gradient
- **Scale**: Logarithmic x-axis for dimension
- **Markers**:
  - MCMC: Square markers with dashed lines
  - HMC: Circle markers with solid lines

## Implementation Notes
- All distributions use NumPy's random number generators
- ESS calculation uses autocorrelation with maximum lag of min(n_samples//3, 50)
- Warmup samples are discarded before computing metrics
- Zero vector used as initial state for all samplers 