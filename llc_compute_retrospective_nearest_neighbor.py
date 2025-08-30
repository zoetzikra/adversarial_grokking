#!/usr/bin/env python3
"""
LLC Retrospective Computation with Nearest-Neighbor Parameters

This script computes LLC for all checkpoints using parameters from the nearest
calibrated checkpoint. This maximizes the value of the hybrid calibration by
using all 18 calibrated parameter sets instead of just 3 fixed ones.

Usage:
    python llc_compute_retrospective_nearest_neighbor.py --checkpoint_dir /path/to/checkpoints --parameter_lookup parameter_lookup.json
"""

import torch
import glob
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import bisect

from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k
from llc_calibrator import LLCCalibrator, LLCCalibratorConfig

def find_nearest_calibrated_params(target_step: int, calibrated_steps: list, calibrated_params: list) -> dict:
    """Find parameters from the nearest calibrated checkpoint"""
    
    # Find the nearest calibrated step
    nearest_idx = min(range(len(calibrated_steps)), 
                     key=lambda i: abs(calibrated_steps[i] - target_step))
    
    nearest_step = calibrated_steps[nearest_idx]
    nearest_params = calibrated_params[nearest_idx]
    
    return nearest_params, nearest_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, 
                       help='Directory containing saved checkpoints')
    parser.add_argument('--parameter_lookup', required=True,
                       help='JSON file with calibrated parameter lookup table')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (default: checkpoint_dir/retrospective_nearest_neighbor)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for LLC estimation')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to use for LLC estimation (0 = use all)')
    parser.add_argument('--device', default='cuda',
                       help='Device to use for computation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output during LLC estimation')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip checkpoints that already have results')
    parser.add_argument('--retrospective_gamma', type=float, default=None,
                       help='Override gamma for retrospective LLC measurements (allows smaller gamma for meaningful variation)')
    parser.add_argument('--use_adversarial_data', action='store_true',
                       help='Generate and use adversarial examples for LLC estimation instead of clean data')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from intermediate results file (e.g., llc_nearest_neighbor_results_intermediate.json or .csv). Skips already processed checkpoints.')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, 'retrospective_nearest_neighbor')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("LLC RETROSPECTIVE COMPUTATION (NEAREST NEIGHBOR)")
    print("=" * 80)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Parameter lookup: {args.parameter_lookup}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    if args.retrospective_gamma is not None:
        print(f"🎯 Retrospective gamma override: {args.retrospective_gamma}")
        print(f"   (Will use this instead of calibrated gamma for LLC measurements)")
    if args.use_adversarial_data:
        print(f"🔥 Adversarial data: ENABLED (will generate PGD adversarial examples)")
        print(f"   Attack parameters: eps=50/255, alpha=4/255, steps=10")
    else:
        print(f"🧼 Adversarial data: DISABLED (using clean training data)")
    if args.resume_from:
        print(f"🔄 Resume mode: ENABLED")
        print(f"   Resume file: {args.resume_from}")
    print("=" * 80)
    
    # Load parameter lookup table
    print("Loading parameter lookup table...")
    with open(args.parameter_lookup, 'r') as f:
        lookup_data = json.load(f)
    
    calibrated_steps = lookup_data['calibrated_steps']
    calibrated_params = lookup_data['calibrated_parameters']
    
    print(f"✅ Loaded parameters for {len(calibrated_steps)} calibrated checkpoints:")
    print(f"   Steps: {calibrated_steps}")
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint-s:*.pt'))
    checkpoint_files = [f for f in checkpoint_files if 'checkpoint-s:-1.pt' not in f]  # Skip base model
    
    # Sort by training step
    def extract_step(filename):
        try:
            return int(filename.split('checkpoint-s:')[1].split('.pt')[0])
        except:
            return -1
    
    checkpoint_files = sorted(checkpoint_files, key=extract_step)
    valid_checkpoints = [(f, extract_step(f)) for f in checkpoint_files if extract_step(f) != -1]
    
    print(f"Found {len(valid_checkpoints)} valid checkpoints to process")
    if len(valid_checkpoints) == 0:
        print("❌ No valid checkpoints found!")
        return
    
    # Load existing results if available
    results_file = os.path.join(args.output_dir, 'llc_nearest_neighbor_results.csv')
    existing_results = {}
    
    # Handle resume functionality
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            print(f"❌ Resume file not found: {args.resume_from}")
            return
        
        print(f"🔄 Resuming from: {args.resume_from}")
        if args.resume_from.endswith('.json'):
            with open(args.resume_from, 'r') as f:
                resume_data = json.load(f)
            # Convert to DataFrame-like format
            existing_results = {int(item['step']): item for item in resume_data}
        elif args.resume_from.endswith('.csv'):
            existing_df = pd.read_csv(args.resume_from)
            existing_results = {int(row['step']): row for _, row in existing_df.iterrows()}
        else:
            print(f"❌ Unsupported resume file format. Use .json or .csv")
            return
        print(f"📊 Resuming with {len(existing_results)} existing results")
        
    elif os.path.exists(results_file) and args.skip_existing:
        existing_df = pd.read_csv(results_file)
        existing_results = {int(row['step']): row for _, row in existing_df.iterrows()}
        print(f"Found {len(existing_results)} existing results")
    
    # Setup model and data
    config = config_resnet18_cifar10()
    config.device = args.device
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Setup data loader
    train_loader, _ = cifar10_dataloaders(config)
    if args.num_samples > 0:
        # Use subset for speed
        subset_indices = torch.randperm(len(train_loader.dataset))[:args.num_samples]
        subset_dataset = torch.utils.data.Subset(train_loader.dataset, subset_indices)
        llc_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Using {args.num_samples} samples for LLC estimation")
    else:
        llc_dataloader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True)
        print(f"Using all {len(train_loader.dataset)} samples for LLC estimation")
    
    # Setup LLC calibrator
    calibrator_config = LLCCalibratorConfig(
        device=args.device,
        batch_size=args.batch_size
    )
    # temporarily override the num_chains to 2
    calibrator_config.num_chains = 2

    llc_calibrator = LLCCalibrator(calibrator_config)

    print(f"\nLLC Estimation Configuration:")
    print(f"  Chains: {calibrator_config.num_chains}")
    print(f"  Steps: {calibrator_config.num_steps}")
    print(f"  Effective draws: {calibrator_config.get_effective_draws()}")
    print(f"  Data samples: {len(llc_dataloader.dataset)}")
    print()
    
    # Compute LLC for each checkpoint
    results = []
    failed_checkpoints = []
    parameter_usage_count = {step: 0 for step in calibrated_steps}
    
    for checkpoint_file, step in tqdm(valid_checkpoints, desc="Computing LLC"):
        # Skip if already computed
        if step in existing_results:
            results.append(existing_results[step])
            continue
        
        try:
            # Find nearest calibrated parameters
            nearest_params, nearest_step = find_nearest_calibrated_params(
                step, calibrated_steps, calibrated_params
            )
            parameter_usage_count[nearest_step] += 1
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            if args.verbose:
                print(f"\nProcessing step {step} (using params from step {nearest_step})...")
                print(f"  ε = {nearest_params['epsilon']:.2e}")
                print(f"  γ = {nearest_params['gamma']:.1f}")
                print(f"  β = {nearest_params['nbeta']:.3f}")
            
            # Compute LLC with nearest neighbor parameters
            llc_results = llc_calibrator.estimate_llc(
                model, 
                train_loader,  # Use full train_loader for consistency
                hyperparams=nearest_params,
                seed=42,  # Fixed seed for reproducibility
                retrospective_gamma=args.retrospective_gamma,  # Override gamma for retrospective measurement
                use_adversarial_data=args.use_adversarial_data  # Use adversarial data if requested
            )
            
            # Extract results
            llc_mean = float(llc_results.get('llc/mean', np.nan))
            
            # Handle llc/stds like llc/means - take mean of the array
            if 'llc/stds' in llc_results and llc_results['llc/stds'] is not None:
                llc_std = float(llc_results['llc/stds'].mean())
            else:
                llc_std = np.nan
            
            # Get MALA acceptance rate if available
            mala_acceptance = float(llc_results.get('mala_accept/mean', np.nan))
            
            result = {
                'step': step,
                'llc_mean': llc_mean,
                'llc_std': llc_std,
                'nearest_calibrated_step': nearest_step,
                'used_epsilon': nearest_params['epsilon'],
                'used_gamma': nearest_params['gamma'],
                'used_nbeta': nearest_params['nbeta'],
                'mala_acceptance': mala_acceptance,
                'checkpoint_file': checkpoint_file,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            results.append(result)
            
            if args.verbose:
                print(f"Step {step}: LLC = {llc_mean:.4f} ± {llc_std:.4f}, MALA = {mala_acceptance:.3f}")
            else:
                print(f"Step {step}: LLC = {llc_mean:.4f} ± {llc_std:.4f} (params from step {nearest_step})")
            
            # Save intermediate results every 5 checkpoints (more frequent to prevent loss on OOM)
            if len(results) % 5 == 0:
                save_results(results, args.output_dir, intermediate=True)
        
        except Exception as e:
            print(f"❌ Failed to process step {step}: {e}")
            failed_checkpoints.append((step, str(e)))
            
            # Add failed result to maintain continuity
            result = {
                'step': step,
                'llc_mean': np.nan,
                'llc_std': np.nan,
                'nearest_calibrated_step': np.nan,
                'used_epsilon': np.nan,
                'used_gamma': np.nan,
                'used_nbeta': np.nan,
                'mala_acceptance': np.nan,
                'checkpoint_file': checkpoint_file,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            results.append(result)
            continue
    
    # Save final results
    save_results(results, args.output_dir, intermediate=False)
    
    # Print parameter usage statistics
    print(f"\n" + "=" * 60)
    print("PARAMETER USAGE STATISTICS")
    print("=" * 60)
    for calib_step in calibrated_steps:
        usage_count = parameter_usage_count[calib_step]
        print(f"Step {calib_step:>6}: Used for {usage_count:>3} checkpoints")
    
    # Print summary
    successful_results = [r for r in results if r.get('success', True)]
    print(f"\n" + "=" * 80)
    print("NEAREST-NEIGHBOR RETROSPECTIVE COMPUTATION COMPLETED")
    print("=" * 80)
    print(f"Total checkpoints processed: {len(results)}")
    print(f"Successful computations: {len(successful_results)}")
    print(f"Failed computations: {len(failed_checkpoints)}")
    print(f"Used parameters from {len(calibrated_steps)} calibrated checkpoints")
    
    if len(successful_results) > 0:
        llc_values = [r['llc_mean'] for r in successful_results if not np.isnan(r['llc_mean'])]
        if len(llc_values) > 0:
            print(f"LLC range: {np.min(llc_values):.4f} - {np.max(llc_values):.4f}")
            print(f"LLC mean: {np.mean(llc_values):.4f} ± {np.std(llc_values):.4f}")
    
    if len(failed_checkpoints) > 0:
        print(f"\nFailed checkpoints:")
        for step, error in failed_checkpoints[:5]:  # Show first 5 failures
            print(f"  Step {step}: {error}")
        if len(failed_checkpoints) > 5:
            print(f"  ... and {len(failed_checkpoints) - 5} more")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 80)

def save_results(results, output_dir, intermediate=False):
    """Save results to CSV and JSON files"""
    
    try:
        if len(results) == 0:
            print("⚠️  No results to save")
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save CSV
        suffix = "_intermediate" if intermediate else ""
        csv_file = os.path.join(output_dir, f'llc_nearest_neighbor_results{suffix}.csv')
        results_df.to_csv(csv_file, index=False)
        
        # Save JSON (more detailed)
        json_file = os.path.join(output_dir, f'llc_nearest_neighbor_results{suffix}.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if not intermediate:
            try:
                # Create summary plot
                create_summary_plot(results_df, output_dir)
            except Exception as e:
                print(f"⚠️  Failed to create summary plot: {e}")
            
            try:
                # Create analysis summary
                create_analysis_summary(results_df, output_dir)
            except Exception as e:
                print(f"⚠️  Failed to create analysis summary: {e}")
        
        print(f"✅ Results saved to {csv_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        # Try to save at least a basic backup
        try:
            backup_file = os.path.join(output_dir, f'backup_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(backup_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📋 Backup saved to {backup_file}")
        except:
            print("❌ Could not save backup either")

def create_summary_plot(df, output_dir):
    """Create summary visualization of LLC evolution"""
    
    # Filter successful results
    successful_df = df[df.get('success', True) == True].copy()
    if len(successful_df) == 0:
        print("⚠️  No successful results to plot")
        return
    
    # Remove NaN values
    plot_df = successful_df.dropna(subset=['llc_mean'])
    if len(plot_df) == 0:
        print("⚠️  No valid LLC values to plot")
        return
    
    # Create plot with single subplot
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    
    # LLC evolution
    ax1.semilogx(plot_df['step'], plot_df['llc_mean'], 'o-', color='blue', linewidth=2, markersize=4)
    if 'llc_std' in plot_df.columns:
        ax1.fill_between(plot_df['step'], 
                        plot_df['llc_mean'] - plot_df['llc_std'],
                        plot_df['llc_mean'] + plot_df['llc_std'],
                        alpha=0.3, color='blue')
    
    # Mark calibrated checkpoints
    calibrated_checkpoints = plot_df[plot_df['step'] == plot_df['nearest_calibrated_step']]
    if len(calibrated_checkpoints) > 0:
        ax1.scatter(calibrated_checkpoints['step'], calibrated_checkpoints['llc_mean'], 
                   color='red', s=100, marker='*', label='Calibrated checkpoints', zorder=5)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Mean Local Learning Coefficient')
    ax1.set_title('LLC Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'llc_nearest_neighbor_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary plot saved to: {plot_path}")

def create_analysis_summary(df, output_dir):
    """Create analysis summary report"""
    
    successful_df = df[df.get('success', True) == True]
    failed_df = df[df.get('success', True) == False]
    
    # Compute statistics
    valid_llc = successful_df.dropna(subset=['llc_mean'])
    
    summary = {
        'total_checkpoints': len(df),
        'successful': len(successful_df),
        'failed': len(failed_df),
        'method': 'nearest_neighbor_parameters'
    }
    
    if len(valid_llc) > 0:
        summary['llc_statistics'] = {
            'count': len(valid_llc),
            'mean': float(valid_llc['llc_mean'].mean()),
            'std': float(valid_llc['llc_mean'].std()),
            'min': float(valid_llc['llc_mean'].min()),
            'max': float(valid_llc['llc_mean'].max()),
            'median': float(valid_llc['llc_mean'].median())
        }
        
        # Parameter usage statistics
        if 'nearest_calibrated_step' in valid_llc.columns:
            param_usage = valid_llc['nearest_calibrated_step'].value_counts().sort_index()
            summary['parameter_usage'] = {
                int(step): int(count) for step, count in param_usage.items() 
                if not np.isnan(step)
            }
    
    # Save summary
    summary_file = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Analysis summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
