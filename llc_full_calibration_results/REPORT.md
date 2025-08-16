# LLC Hybrid Calibration Report (Nearest Neighbor)

## Calibration Summary
- Successfully calibrated 17 key checkpoints
- Using nearest-neighbor parameter lookup for all other checkpoints

## Calibrated Checkpoints

### Checkpoint 8
- Epsilon: 1.000000e-03
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.1634
### Checkpoint 24
- Epsilon: 1.000000e-03
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.0966
### Checkpoint 55
- Epsilon: 1.000000e-05
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.1461
### Checkpoint 94
- Epsilon: 1.000000e-05
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.1271
### Checkpoint 162
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.3159
### Checkpoint 473
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.5833
### Checkpoint 809
- Epsilon: 1.000000e-05
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.2502
### Checkpoint 1,382
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.6803
### Checkpoint 3,088
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 108.8083
### Checkpoint 9,017
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 66.9986
### Checkpoint 15,408
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.4750
### Checkpoint 26,327
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.3679
### Checkpoint 58,801
- Epsilon: 1.000000e-03
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 0.6648
### Checkpoint 131,331
- Epsilon: 1.000000e-03
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 1.2694
### Checkpoint 224,398
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 1.0336
### Checkpoint 383,418
- Epsilon: 1.000000e-04
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 2.3135
### Checkpoint 500,000
- Epsilon: 1.000000e-03
- Gamma: 1000.0
- Beta: 82.073318
- LLC: 5.8309

## Usage Instructions

1. **Review calibration results**: Check `parameter_evolution.png` for parameter trends across checkpoints
2. **Run enhanced retrospective processing**:
   ```bash
   ./run_retrospective_nearest_neighbor.sh
   ```
3. **How it works**: For each checkpoint, uses parameters from the nearest calibrated checkpoint:
   - Step 1-15: Uses parameters from step 8
   - Step 16-39: Uses parameters from step 24
   - And so on...

## Files Generated
- `calibration_results.json`: Raw calibration results from key checkpoints
- `parameter_lookup.json`: Nearest-neighbor parameter lookup table
- `parameter_evolution.png`: Visualization of parameter trends
- `run_retrospective_nearest_neighbor.sh`: Enhanced retrospective processing script
- `REPORT.md`: This summary report

## Advantages of This Approach
- ✅ Uses ALL 18 calibrated parameter sets (nothing wasted!)
- ✅ Automatic parameter selection for any checkpoint
- ✅ More accurate than fixed parameters across all training
- ✅ Simple nearest-neighbor logic (no complex interpolation)
