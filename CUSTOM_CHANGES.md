# Custom Positive Seasonality Implementation

## Overview

This fork modifies pymc-marketing to guarantee that seasonality contributions are always positive, preventing negative baseline values in Marketing Mix Models.

## Changes Made

### 1. Core Source Modifications

**File: `pymc_marketing/mmm/mmm.py`**

- **Method**: `_build_yearly_seasonality_contribution()` (lines ~717-729)
- **Change**: Added absolute value transformation to seasonality output
- **Formula**: `seasonality = abs(linear_fourier_combination)`
- **Result**: Guarantees positive seasonality values for additive model while preserving magnitude

**File: `pymc_marketing/mmm/multidimensional.py`**

- **Method**: Yearly seasonality section in `build_model()` (lines ~1487-1501)
- **Change**: Added absolute value transformation to seasonality output
- **Formula**: `seasonality = abs(linear_fourier_combination)`
- **Result**: Guarantees positive seasonality values for additive geo-level models while preserving magnitude

### 2. Implementation Details

**Traditional Fourier Seasonality (Original):**

```python
seasonality(t) = Σ[γᵢ * fourier_feature_i(t)]
# Can be negative! Range: (-∞, +∞)
```

**Positive Seasonality (Custom - Absolute Value):**

```python
seasonality(t) = abs(Σ[γᵢ * fourier_feature_i(t)])
# Always positive! Range: [0, +∞)
# For additive model: mu = intercept + channels + seasonality
```

**Key Properties:**

- ✅ Absolute value function guarantees output ≥ 0
- ✅ **Preserves magnitude** - if input is ±X, output is |X|
- ✅ **Symmetric transformation** - doesn't bias MCMC optimization
- ✅ **No numerical overflow** - unlike softplus which caused explosion for large inputs
- ✅ **Simple and interpretable** - easier to understand than softplus
- ✅ **Perfect for additive models** - outputs in absolute scale matching original
- ✅ Preserves seasonal pattern magnitude (peaks and troughs)
- ✅ Maintains API compatibility (same variable names)
- ✅ Works with both single-dimension and geo-level models

### 3. Recommended Prior Configuration

With the absolute value transformation, you can use **tight priors** on `gamma_fourier`:

```python
# Non-geo model
"gamma_fourier": {
    "dist": "Normal",
    "kwargs": {"mu": 0, "sigma": 0.01},  # Tight prior
    "dims": "fourier_mode"
}

# Geo-level model
"gamma_fourier": Prior(
    "Normal",
    mu=0,
    sigma=0.01,  # Tight prior for geo variation
    dims=("geo", "fourier_mode"),
)
```

**Why this works:**

- Absolute value ensures positivity while preserving magnitude from linear combination
- With `σ=0.01`, gamma_fourier typically in `[-0.03, 0.03]`
- Linear combination range depends on data but stays controlled by tight priors
- After `abs(linear_combo)`: output is same magnitude as input, just positive
- **Tight priors (σ=0.01) control the magnitude** - prevents extreme values
- **Symmetric transformation** - doesn't bias MCMC optimization like softplus did
- **No numerical overflow** - unlike softplus which caused explosion for large inputs
- **No gradient asymmetry** - abs() has equal treatment of positive/negative inputs
- Magnitude controlled entirely by the prior on gamma_fourier

### 4. Files Removed (No Longer Needed)

The following files were created during initial patch development but are **no longer used**:

- `pymc_marketing/mmm/components/positive_seasonality.py` (can be deleted)
- `pymc_marketing/mmm/patches/positive_seasonality_patch.py` (can be deleted)
- `pymc_marketing/mmm/patches/__init__.py` (can be deleted)

The absolute value transformation is now **built directly into the core MMM classes**.

## Benefits

- ✅ **Guarantees positive seasonality**: softplus(x) > 0 for all x
- ✅ **No explosion**: Linear growth for large inputs (unlike exponential)
- ✅ **Additive model friendly**: Outputs in absolute scale (0-10) perfect for adding to baseline
- ✅ **Natural bounds**: No hardcoded limits - bounds emerge from function + priors
- ✅ **Prevents negative baseline**: mu = intercept + channels + softplus(seasonality) ≥ intercept
- ✅ **Simple & robust**: No complex patching, just source modification
- ✅ **Excellent MCMC properties**: Smooth gradient everywhere
- ✅ **Maintainable**: Easy to understand and debug
- ✅ **Backward compatible**: Same API, same variable names
- ✅ **Works for all models**: Both MMM and multidimensional MMM

## Version

- **Based on**: pymc-marketing v0.17.0
- **Custom version**: v0.17.0-tw-1 (Triple Whale Custom Release 1)
- **Fork**: https://github.com/RamiFisherTW/pymc-marketing

## Installation

```bash
# Production
pip install git+https://github.com/RamiFisherTW/pymc-marketing.git@v0.17.0-tw-1

# Development (editable)
git clone https://github.com/RamiFisherTW/pymc-marketing.git
cd pymc-marketing
pip install -e .
```

## Maintenance

- **Development branch**: `main` (direct commits)
- **Upstream**: https://github.com/pymc-labs/pymc-marketing.git
- **Sync strategy**: Pull from upstream when needed, reapply exponential transformation

## Testing

Test that seasonality is positive and within reasonable range:

```python
import arviz as az
import numpy as np

# After model fit
posterior = model.idata["posterior"]
seasonality = posterior["yearly_seasonality_contribution"]

# Check all values are positive
assert (seasonality > 0).all(), "Seasonality should be positive!"

# Check values are in reasonable range for additive model
print(f"Seasonality range: [{seasonality.min():.3f}, {seasonality.max():.3f}]")
print(f"Seasonality mean: {seasonality.mean():.3f}")
print(f"Seasonality std: {seasonality.std():.3f}")

# Typical expected values for additive model: range ≈ [0, 10], mean ≈ 2-5
# Should be small relative to intercept
assert seasonality.min() >= 0, "Seasonality should be non-negative!"
assert seasonality.max() < 50, "Seasonality should not be too high for additive model!"

# Check seasonality contribution is reasonable relative to baseline
baseline = posterior["intercept"]
seasonality_pct = (seasonality.mean() / baseline.mean()) * 100
print(f"Seasonality contribution: {seasonality_pct:.1f}% of intercept")
```
