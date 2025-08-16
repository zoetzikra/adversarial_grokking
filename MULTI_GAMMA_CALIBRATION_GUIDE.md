# Multi-Gamma LLC Calibration Guide

## Overview

The new multi-gamma calibration feature allows you to jointly optimize `(epsilon, gamma, beta)` hyperparameters by testing multiple gamma values and selecting the combination that achieves the best MALA acceptance rate.

## Why This Matters

### Problems with Single-Gamma Calibration:
- **Gamma is arbitrarily fixed** - usually to a single value like 5000
- **Optimal epsilon depends on gamma** - different temperatures require different step sizes
- **Suboptimal sampling quality** - may not achieve ideal MALA acceptance rates (~0.92)
- **High LLC variance** - poor sampling leads to unstable LLC estimates

### Benefits of Multi-Gamma Calibration:
- **Joint optimization** - finds the best (ε, γ, β) combination globally
- **Better sampling quality** - targets optimal MALA acceptance rates
- **More stable LLC estimates** - better sampling reduces variance
- **Principled selection** - uses sampling diagnostics rather than arbitrary choices

## Usage

### Basic Usage

```python
from llc_calibrator import LLCCalibrator, LLCCalibratorConfig

# Setup calibrator
calibrator_config = LLCCalibratorConfig(device='cuda')
calibrator_config.calibration_epsilons = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
llc_calibrator = LLCCalibrator(calibrator_config)

# Multi-gamma calibration
gamma_values = [1000.0, 5000.0, 10000.0]
optimal_params = llc_calibrator.calibrate_hyperparameters_multi_gamma(
    model=model,
    train_loader=train_loader,
    gamma_values=gamma_values,
    checkpoint_name="my_model",
    use_tuned_beta=False
)

print(f"Best combination:")
print(f"  ε = {optimal_params['epsilon']:.2e}")
print(f"  γ = {optimal_params['gamma']:.0f}")
print(f"  β = {optimal_params['nbeta']:.3f}")
print(f"  MALA acceptance = {optimal_params['mala_acceptance']:.3f}")
```

### Advanced Usage

```python
# Test more gamma values for thorough search
gamma_values = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]

# Use tuned beta values from EpsilonBetaAnalyzer
optimal_params = llc_calibrator.calibrate_hyperparameters_multi_gamma(
    model=model,
    train_loader=train_loader,
    gamma_values=gamma_values,
    checkpoint_name="thorough_search",
    use_tuned_beta=True  # Use analyzer-tuned beta instead of default
)
```

## How It Works

1. **Run calibration for each gamma value**
   - For each γ in `gamma_values`, run full epsilon-beta sweep
   - Each sweep finds the best (ε, β) for that specific γ

2. **Collect MALA acceptance rates**
   - Extract MALA acceptance rate for each (ε, γ, β) combination
   - Target acceptance rate is 0.92 (ideal for SGLD sampling)

3. **Global selection**
   - Compare all combinations across all gamma values
   - Select the (ε, γ, β) with MALA acceptance closest to 0.92
   - Display ranked comparison of all results

## Output Example

```
============================================================
MULTI-GAMMA RESULT COMPARISON
============================================================
Results ranked by MALA acceptance (target = 0.920):
--------------------------------------------------------------------------------
🏆 γ= 10000, ε=5.00e-06, β= 82.1, MALA=0.915 (Δ=0.005), LLC=0.234±1.245
 2. γ=  5000, ε=1.14e-05, β= 82.1, MALA=0.555 (Δ=0.365), LLC=0.214±4.625
 3. γ=  1000, ε=1.00e-05, β= 82.1, MALA=0.425 (Δ=0.495), LLC=0.189±6.234

🏆 SELECTED BEST COMBINATION:
   γ = 10000.0
   ε = 5.00e-06
   β = 82.073
   MALA acceptance = 0.915
   LLC mean = 0.234
============================================================
```

## Recommended Gamma Ranges

### For different model sizes:
- **Small models** (< 1M params): `[1000, 5000, 10000]`
- **Medium models** (1M-10M params): `[2000, 5000, 10000, 20000]`
- **Large models** (> 10M params): `[5000, 10000, 20000, 50000]`

### For different training stages:
- **Well-converged models**: `[1000, 5000, 10000]`
- **Partially trained models**: `[5000, 10000, 20000]`
- **Early training**: `[10000, 20000, 50000]`

## Interpreting Results

### MALA Acceptance Rate:
- **0.90-0.95**: Excellent sampling quality ✅
- **0.80-0.90**: Good sampling quality ✅
- **0.60-0.80**: Acceptable but suboptimal ⚠️
- **< 0.60**: Poor sampling quality ❌

### LLC Statistics:
- **Positive LLC mean**: Good ✅
- **LLC std/mean < 2**: Stable sampling ✅
- **LLC std/mean > 5**: High variance, consider different parameters ⚠️

## Troubleshooting

### If all gamma values fail:
- Check model convergence
- Reduce epsilon range (try smaller values)
- Increase gamma range (try larger values)
- Verify model is in eval mode with gradients enabled

### If MALA acceptance is consistently low:
- Try larger gamma values
- Reduce epsilon values
- Check for in-place operations in model
- Verify model parameters require gradients

### If LLC variance is high:
- Increase gamma (higher temperature = more exploration)
- Reduce epsilon (smaller steps = more stable)
- Increase number of SGLD draws (longer chains)

## Files

- `test_multi_gamma_calibration.py` - Standalone test script
- `test_mala_calibration.py` - Updated to compare single vs multi-gamma
- `llc_calibrator.py` - Contains the new `calibrate_hyperparameters_multi_gamma()` method

## Backward Compatibility

The original `calibrate_hyperparameters()` method remains unchanged and fully functional. You can gradually migrate to multi-gamma calibration or use both approaches for comparison.
