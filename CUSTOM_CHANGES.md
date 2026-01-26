# Custom Positive Seasonality Implementation

## Overview

This fork modifies pymc-marketing to guarantee that seasonality contributions are always positive, preventing negative baseline values in Marketing Mix Models.

## Changes Made

### 1. Core Source Modifications

**File: `pymc_marketing/mmm/mmm.py`**

- **Method**: `_build_yearly_seasonality_contribution()` (lines ~717-729)
- **Change**: Added scaled exponential transformation to seasonality output
- **Formula**: `seasonality = exp(0.1 * linear_fourier_combination)`
- **Result**: Guarantees positive seasonality values with controlled magnitude

**File: `pymc_marketing/mmm/multidimensional.py`**

- **Method**: Yearly seasonality section in `build_model()` (lines ~1487-1499)
- **Change**: Added scaled exponential transformation to seasonality output
- **Formula**: `seasonality = exp(0.1 * linear_fourier_combination)`
- **Result**: Guarantees positive seasonality values with controlled magnitude for geo-level models

### 2. Implementation Details

**Traditional Fourier Seasonality (Original):**

```python
seasonality(t) = Σ[γᵢ * fourier_feature_i(t)]
# Can be negative! Range: (-∞, +∞)
```

**Positive Seasonality (Custom):**

```python
scale_factor = 0.1
seasonality(t) = exp(scale_factor * Σ[γᵢ * fourier_feature_i(t)])
# Always positive! Range: (0, +∞) with typical values in [0.74, 1.35]
```

**Key Properties:**

- ✅ Exponential function guarantees output > 0
- ✅ Scale factor (0.1) controls magnitude, preventing explosion
- ✅ Centered around 1.0 (multiplicative effect)
- ✅ Preserves seasonal pattern (peaks and troughs)
- ✅ Maintains API compatibility (same variable names)
- ✅ Works with both single-dimension and geo-level models

### 3. Recommended Prior Configuration

With the scaled exponential transformation, you can use **standard priors** on `gamma_fourier`:

```python
# Non-geo model
"gamma_fourier": {
    "dist": "Normal",
    "kwargs": {"mu": 0, "sigma": 1.0},  # Standard normal
    "dims": "fourier_mode"
}

# Geo-level model
"gamma_fourier": Prior(
    "Normal",
    mu=0,
    sigma=1.0,  # Standard normal for geo variation
    dims=("geo", "fourier_mode"),
)
```

**Why this works:**

- Scale factor (0.1) controls the magnitude, not the prior
- With `σ=1.0`, linear_combo typically in `[-3, 3]`
- After `exp(0.1 * linear_combo)`: seasonality range ≈ [0.74, 1.35] (~±30% variation)
- Provides reasonable seasonal patterns without explosion
- More flexible than overly tight priors

### 4. Files Removed (No Longer Needed)

The following files were created during initial patch development but are **no longer used**:

- `pymc_marketing/mmm/components/positive_seasonality.py` (can be deleted)
- `pymc_marketing/mmm/patches/positive_seasonality_patch.py` (can be deleted)
- `pymc_marketing/mmm/patches/__init__.py` (can be deleted)

The exponential transformation is now **built directly into the core MMM classes**.

## Benefits

- ✅ **Prevents negative baseline**: baseline = intercept * seasonality > 0
- ✅ **Controlled magnitude**: Scale factor prevents exponential explosion
- ✅ **Multiplicative effect**: Seasonality centered at 1.0 (natural interpretation)
- ✅ **Simple & robust**: No complex patching, just source modification
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

# Check values are in reasonable range
print(f"Seasonality range: [{seasonality.min():.3f}, {seasonality.max():.3f}]")
print(f"Seasonality mean: {seasonality.mean():.3f}")
print(f"Seasonality std: {seasonality.std():.3f}")

# Typical expected values: mean ≈ 1.0, range ≈ [0.7, 1.4]
assert seasonality.max() < 5, "Seasonality should not be too high!"
```
