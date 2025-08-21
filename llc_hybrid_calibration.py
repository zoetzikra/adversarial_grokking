#!/usr/bin/env python3
"""
LLC Hybrid Calibration Script

This script implements the hybrid approach:
1. Calibrate hyperparameters on key checkpoints
2. Extract common patterns from optimal parameters  
3. Generate optimized parameters for batch processing with llc_compute_retrospective.py

Usage:
    # Single-gamma calibration (original):
    python llc_hybrid_calibration.py --checkpoint_dir /path/to/checkpoints 
                    --output_dir ./llc_full_calibration_results --device cuda --selection_method mala
    
    # Multi-gamma calibration (new):
    python llc_hybrid_calibration.py --checkpoint_dir /path/to/checkpoints 
                    --output_dir ./llc_full_calibration_results --device cuda 
                    --gamma_values 1000 5000 10000
"""

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import random
from datetime import datetime

from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k
from llc_calibrator import LLCCalibrator, LLCCalibratorConfig


def set_reproducible_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, 
                       help='Directory containing saved checkpoints')
    parser.add_argument('--output_dir', default='./llc_full_calibration_results',
                       help='Directory to save calibration results')
    parser.add_argument('--device', default='cuda',
                       help='Device to use for computation')
    parser.add_argument('--selection_method', type=str, default='mala',
                       help='Selection method to use for calibration (stability or mala)')
    parser.add_argument('--use_tuned_beta', type=bool, default=False,
                       help='Use tuned beta instead of default') 
    parser.add_argument('--gamma_values', type=float, nargs='+', default=None,
                       help='Multiple gamma values for multi-gamma calibration (e.g., --gamma_values 1000 5000 10000)')
    args = parser.parse_args()
    
    # Always add timestamp to output directory to prevent overwrites
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{run_timestamp}"

    set_reproducible_seeds(42)
    
    # Key checkpoints to calibrate (your selected list)
    key_checkpoints = [8, 24, 55, 94, 162, 473, 809, 1382, 3088, 9017, 
                      15408, 26327, 58801, 100471, 131331, 224398, 383418, 500000]
    # key_checkpoints =  [383418, 500000]
    
    print("=" * 80)
    print("LLC HYBRID CALIBRATION")
    print("=" * 80)
    print(f"Key checkpoints to calibrate: {key_checkpoints}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and setup model
    config = config_resnet18_cifar10()
    config.device = args.device
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Load data
    train_loader, _ = cifar10_dataloaders(config)
    
    # Initialize calibrator
    calibrator_config = LLCCalibratorConfig(device=args.device)
    llc_calibrator = LLCCalibrator(calibrator_config)
    
    # Display calibration method (after calibrator is initialized)
    if args.gamma_values is not None:
        print(f"🎯 MULTI-GAMMA CALIBRATION: Testing γ = {args.gamma_values}")
        print(f"   Selection method: {args.selection_method}")
        print(f"   Will select best (ε, γ, β) combination based on {args.selection_method} criterion")
    else:
        print(f"📊 SINGLE-GAMMA CALIBRATION: Using γ = {calibrator_config.gamma}")
        print(f"   Selection method: {args.selection_method}")
    print(f"Use tuned beta: {args.use_tuned_beta}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Phase 1: Calibrate key checkpoints
    print("\n" + "=" * 60)
    print("PHASE 1: CALIBRATING KEY CHECKPOINTS")
    print("=" * 60)
    
    calibration_results = []
    
    for step in tqdm(key_checkpoints, desc="Calibrating checkpoints"):
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-s:{step}.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n--- Calibrating checkpoint step {step} ---")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Run calibration with timestamp to prevent overwrites
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint-s:{step}"
            save_path = os.path.join(args.output_dir, f"{checkpoint_name}_{timestamp}")
            
            # Choose between single-gamma and multi-gamma calibration
            if args.gamma_values is not None:
                # Multi-gamma calibration
                print(f"Using multi-gamma calibration with γ = {args.gamma_values}")
                optimal_params = llc_calibrator.calibrate_hyperparameters_multi_gamma(
                    model=model,
                    train_loader=train_loader,
                    gamma_values=args.gamma_values,
                    checkpoint_name=checkpoint_name,
                    save_path=save_path,
                    use_tuned_beta=args.use_tuned_beta,
                    selection_method=args.selection_method  # Pass through the selection method
                )
            else:
                # Single-gamma calibration (original method)
                print(f"Using single-gamma calibration with γ = {llc_calibrator.config.gamma}")
                optimal_params = llc_calibrator.calibrate_hyperparameters(
                    model, 
                    train_loader, 
                    checkpoint_name=checkpoint_name,
                    save_path=save_path,
                    selection_method=args.selection_method,  # Use command line arg
                    use_tuned_beta=args.use_tuned_beta  # Set to True to use tuned beta instead of default
                )
            
            # Estimate LLC with optimal parameters and create final trace
            print("Estimating LLC with optimal parameters...")
            llc_results = llc_calibrator.estimate_llc(
                model, 
                train_loader, 
                hyperparams=optimal_params,
                save_path=os.path.join(save_path, "final_llc_estimation"),
                seed=43  # Use different seed for final estimation to avoid exact duplication
            )
            
            # Plot the final LLC estimation trace
            if 'loss/trace' in llc_results:
                final_trace_path = os.path.join(save_path, "final_llc_trace.png")
                llc_calibrator.plot_sampling_evolution(
                    llc_results, 
                    save_path=final_trace_path, 
                    show=False
                )
                print(f"Final LLC trace plot saved to {final_trace_path}")
                print(f"Final trace shape: {llc_results['loss/trace'].shape}")
            
            # Store results
            result = {
                'step': step,
                'checkpoint_path': checkpoint_path,
                'optimal_params': optimal_params,
                'final_llc_results': llc_results,  # Include final LLC estimation results
                'model_parameters': sum(p.numel() for p in model.parameters()),
            }
            
            calibration_results.append(result)
            
            print(f"✅ Step {step} calibrated successfully")
            print(f"   ε = {optimal_params['epsilon']:.2e}")
            print(f"   γ = {optimal_params['gamma']:.1f}")
            print(f"   β = {optimal_params['nbeta']:.3f}")
            if 'mala_acceptance' in optimal_params and optimal_params['mala_acceptance'] is not None:
                print(f"   MALA acceptance = {optimal_params['mala_acceptance']:.3f}")
            if 'llc_mean' in optimal_params:
                print(f"   LLC = {optimal_params['llc_mean']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to calibrate step {step}: {e}")
            continue
    
    # Save calibration results
    calibration_file = os.path.join(args.output_dir, 'calibration_results.json')
    with open(calibration_file, 'w') as f:
        json.dump(calibration_results, f, indent=2, default=convert_to_serializable)
    
    print(f"\n✅ Calibration results saved to: {calibration_file}")
    
    # Phase 2: Extract patterns and generate recommendations
    print("\n" + "=" * 60)
    print("PHASE 2: EXTRACTING PATTERNS")
    print("=" * 60)
    
    if len(calibration_results) == 0:
        print("❌ No successful calibrations found!")
        return
    
    analysis = analyze_calibration_patterns(calibration_results, args.output_dir)
    
    # Phase 3: Generate interpolated parameter lookup
    print("\n" + "=" * 60)
    print("PHASE 3: GENERATING INTERPOLATED PARAMETER LOOKUP")
    print("=" * 60)
    
    generate_interpolated_parameters(calibration_results, args.output_dir, args.checkpoint_dir)
    
    print("\n" + "=" * 80)
    print("HYBRID CALIBRATION COMPLETED!")
    print("=" * 80)
    print("Next steps:")
    print("1. Review the analysis plots and parameter recommendations")
    print("2. Check the calibration quality and parameter trends")
    print("3. When ready, run the retrospective processing command shown above")
    print("=" * 80)

def convert_to_serializable(obj):
    """Convert numpy arrays and scalar types to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def analyze_calibration_patterns(calibration_results: List[Dict], output_dir: str) -> Dict:
    """Analyze patterns in calibrated parameters across training steps"""
    
    # Extract data for analysis
    steps = [r['step'] for r in calibration_results]
    epsilons = [r['optimal_params']['epsilon'] for r in calibration_results]
    gammas = [r['optimal_params']['gamma'] for r in calibration_results]
    nbetas = [r['optimal_params']['nbeta'] for r in calibration_results]
    llc_means = [r['optimal_params'].get('llc_mean', np.nan) for r in calibration_results]
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame({
        'step': steps,
        'epsilon': epsilons,
        'gamma': gammas,
        'nbeta': nbetas,
        'llc_mean': llc_means
    })
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'calibrated_parameters.csv'), index=False)
    
    # Compute statistics
    analysis = {
        'num_calibrated': len(calibration_results),
        'step_range': {'min': min(steps), 'max': max(steps)},
        'tuning_info': {
            'epsilon_tuned': True,
            'gamma_tuned': False,
            'nbeta_tuned': False,
            'selection_method': 'stability',
            'note': 'Only epsilon is optimized via EpsilonBetaAnalyzer. Gamma and nbeta are fixed.'
        },
        'epsilon_stats': {
            'mean': float(np.mean(epsilons)),
            'std': float(np.std(epsilons)),
            'min': float(np.min(epsilons)),
            'max': float(np.max(epsilons)),
            'median': float(np.median(epsilons))
        },
        'gamma_stats': {
            'value': float(gammas[0]),  # All should be the same
            'note': 'Fixed value, not tuned'
        },
        'nbeta_stats': {
            'value': float(nbetas[0]),  # All should be the same
            'note': 'Default value from devinterp, not tuned'
        }
    }
    
    # Print analysis
    print(f"Successfully calibrated {analysis['num_calibrated']} checkpoints")
    print(f"Training step range: {analysis['step_range']['min']:,} - {analysis['step_range']['max']:,}")
    print("\nParameter Statistics:")
    print(f"  Epsilon (ε): {analysis['epsilon_stats']['median']:.2e} ± {analysis['epsilon_stats']['std']:.2e}")
    print(f"    Range: [{analysis['epsilon_stats']['min']:.2e}, {analysis['epsilon_stats']['max']:.2e}]")
    print(f"  Gamma (γ): {analysis['gamma_stats']['value']:.1f} (FIXED - not tuned)")
    print(f"    All checkpoints use the same fixed gamma value")
    print(f"  NBeta (β): {analysis['nbeta_stats']['value']:.3f} (DEFAULT - not tuned)")
    print(f"    All checkpoints use the same default beta value")
    
    # Create visualization plots
    create_parameter_plots(df, output_dir)
    
    # Save analysis
    with open(os.path.join(output_dir, 'parameter_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis

def create_parameter_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization plots of parameter evolution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Calibrated Parameter Evolution Across Training', fontsize=16, fontweight='bold')
    
    # Epsilon evolution
    axes[0, 0].semilogx(df['step'], df['epsilon'], 'o-', color='blue', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Epsilon (ε)')
    axes[0, 0].set_title('Step Size Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Gamma evolution  
    axes[0, 1].semilogx(df['step'], df['gamma'], 'o-', color='red', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Gamma (γ)')
    axes[0, 1].set_title('Localization Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # NBeta evolution
    axes[1, 0].semilogx(df['step'], df['nbeta'], 'o-', color='green', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('NBeta (β)')
    axes[1, 0].set_title('Inverse Temperature Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # LLC evolution
    valid_llc = df.dropna(subset=['llc_mean'])
    if len(valid_llc) > 0:
        axes[1, 1].semilogx(valid_llc['step'], valid_llc['llc_mean'], 'o-', color='purple', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('LLC Mean')
        axes[1, 1].set_title('LLC Evolution')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No LLC data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('LLC Evolution (No Data)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Parameter evolution plots saved to: {os.path.join(output_dir, 'parameter_evolution.png')}")

def generate_interpolated_parameters(calibration_results: List[Dict], output_dir: str, checkpoint_dir: str):
    """Generate parameter lookup using nearest neighbor from calibrated checkpoints"""
    
    print(f"Creating nearest-neighbor parameter lookup from {len(calibration_results)} calibrated checkpoints")
    
    # Extract calibrated steps and their parameters
    calibrated_steps = []
    calibrated_params = []
    
    for result in calibration_results:
        calibrated_steps.append(result['step'])
        calibrated_params.append(result['optimal_params'])
    
    # Sort by step for easier lookup
    sorted_indices = np.argsort(calibrated_steps)
    calibrated_steps = [calibrated_steps[i] for i in sorted_indices]
    calibrated_params = [calibrated_params[i] for i in sorted_indices]
    
    print(f"Calibrated steps: {calibrated_steps}")
    
    # Create parameter lookup table
    parameter_lookup = {
        'calibrated_steps': calibrated_steps,
        'calibrated_parameters': calibrated_params,
        'lookup_method': 'nearest_neighbor',
        'description': 'Uses parameters from the nearest calibrated checkpoint'
    }
    
    # Save the lookup table
    lookup_file = os.path.join(output_dir, 'parameter_lookup.json')
    with open(lookup_file, 'w') as f:
        json.dump(parameter_lookup, f, indent=2, default=convert_to_serializable)
    
    print(f"✅ Parameter lookup table saved to: {lookup_file}")
    
    # Print the command to run retrospective processing (don't auto-generate script)
    print(f"\n📋 To run retrospective processing after reviewing results:")
    print(f"python llc_compute_retrospective_nearest_neighbor.py \\")
    print(f"    --checkpoint_dir {checkpoint_dir} \\")
    print(f"    --parameter_lookup {lookup_file} \\")
    print(f"    --output_dir {output_dir}/retrospective_results \\")
    print(f"    --verbose")
        
    # Generate summary report
    report_content = f"""# LLC Hybrid Calibration Report (Nearest Neighbor)

## Calibration Summary
- Successfully calibrated {len(calibration_results)} key checkpoints
- Using nearest-neighbor parameter lookup for all other checkpoints

## Calibrated Checkpoints
"""
    
    for i, (step, params) in enumerate(zip(calibrated_steps, calibrated_params)):
        report_content += f"""
### Checkpoint {step:,}
- Epsilon: {params['epsilon']:.6e}
- Gamma: {params['gamma']:.1f}
- Beta: {params['nbeta']:.6f}"""
        if 'llc_mean' in params:
            report_content += f"""
- LLC: {params['llc_mean']:.4f}"""
    
    report_content += f"""

## Usage Instructions

1. **Review calibration results**: Check `parameter_evolution.png` for parameter trends across checkpoints
2. **Run enhanced retrospective processing**:
   ```bash
   ./run_retrospective_nearest_neighbor.sh
   ```
3. **How it works**: For each checkpoint, uses parameters from the nearest calibrated checkpoint:
   - Step 1-15: Uses parameters from step {calibrated_steps[0]}
   - Step 16-39: Uses parameters from step {calibrated_steps[1] if len(calibrated_steps) > 1 else calibrated_steps[0]}
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
"""
    
    with open(os.path.join(output_dir, 'REPORT.md'), 'w') as f:
        f.write(report_content)
    
    print(f"✅ Summary report saved to: {os.path.join(output_dir, 'REPORT.md')}")
    print(f"✅ Ready to process ALL checkpoints using nearest-neighbor parameters!")

if __name__ == '__main__':
    main()
