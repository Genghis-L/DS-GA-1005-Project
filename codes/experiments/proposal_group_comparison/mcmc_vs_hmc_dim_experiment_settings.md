# Experiment Settings for Proposal Group Comparison

## General Settings
| Parameter | Value |
|-----------|-------|
| Number of Points to Sample | 1000 |
| Warmup Samples | 100 |
| Dimensions Tested | [2, 5, 10, 20, 50, 100, 200, 500] |

## Target Distributions

### Standard Gaussian
| Parameter | Value |
|-----------|-------|
| Type | Multivariate Normal |
| Mean | Zero vector |
| Covariance | Identity matrix |

### Donut Distribution
| Parameter | Value |
|-----------|-------|
| Type | N-dimensional shell |
| Radius | 2.0 |
| Width | 0.5 |

## MCMC Settings

### Proposal Distributions
| Type | Parameters |
|------|------------|
| Gaussian | Covariance = Identity matrix |
| Student-t | Degrees of freedom = 3 |
| Uniform | Range = [-1, 1] |

## HMC Settings

### Leapfrog Integrator Parameters
| Parameter | Value |
|-----------|-------|
| Number of Steps (L) | 50 |
| Step Size (ε) | 0.1 |

### Kinetic Energy Distributions
| Type | Parameters |
|------|------------|
| Gaussian | Standard normal momentum |
| Student-t | Degrees of freedom (ν) = 3.0 |
| Uniform | Scale = 1.0 |

## Performance Metrics
- Acceptance Rate
- Effective Sample Size (ESS)
- Computation Time

## Implementation Details
- All distributions are implemented with NumPy for efficient matrix operations
- ESS calculation uses autocorrelation with maximum lag of min(n_samples//3, 100)
- Warmup samples are discarded before computing metrics 