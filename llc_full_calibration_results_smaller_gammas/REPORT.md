# LLC Hybrid Calibration Report (Nearest Neighbor)

## Calibration Summary
- Successfully calibrated 18 key checkpoints
- Using nearest-neighbor parameter lookup for all other checkpoints

## Calibrated Checkpoints

### Checkpoint 8
- Epsilon: 1.000000e-05
- Gamma: 10.0
- Beta: 0.820733
- LLC: 0.1006
### Checkpoint 24
- Epsilon: 1.000000e-06
- Gamma: 10.0
- Beta: 0.820733
- LLC: 0.0105
### Checkpoint 55
- Epsilon: 1.000000e-06
- Gamma: 50.0
- Beta: 0.820733
- LLC: 0.0495
### Checkpoint 94
- Epsilon: 1.000000e-06
- Gamma: 50.0
- Beta: 0.820733
- LLC: 0.0187
### Checkpoint 162
- Epsilon: 1.000000e-05
- Gamma: 10.0
- Beta: 0.820733
- LLC: 0.5720
### Checkpoint 473
- Epsilon: 1.000000e-06
- Gamma: 10.0
- Beta: 0.820733
- LLC: 0.0436
### Checkpoint 809
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0433
### Checkpoint 1,382
- Epsilon: 1.000000e-06
- Gamma: 100.0
- Beta: 0.820733
- LLC: 0.0392
### Checkpoint 3,088
- Epsilon: 1.000000e-06
- Gamma: 100.0
- Beta: 0.820733
- LLC: 0.0226
### Checkpoint 9,017
- Epsilon: 1.000000e-06
- Gamma: 100.0
- Beta: 0.820733
- LLC: 0.0218
### Checkpoint 15,408
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0334
### Checkpoint 26,327
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0197
### Checkpoint 58,801
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0031
### Checkpoint 100,471
- Epsilon: 1.000000e-06
- Gamma: 10.0
- Beta: 0.820733
- LLC: 0.0222
### Checkpoint 131,331
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0208
### Checkpoint 224,398
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0347
### Checkpoint 383,418
- Epsilon: 1.000000e-06
- Gamma: 200.0
- Beta: 0.820733
- LLC: 0.0733
### Checkpoint 500,000
- Epsilon: 1.000000e-06
- Gamma: 10.0
- Beta: 0.820733
- LLC: 0.0007

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
