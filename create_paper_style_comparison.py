#!/usr/bin/env python3
"""
Create paper-style comparison plots between Local Complexity and LLC metrics
Reproduces the style from the "Deep Networks Always Grok" paper for ResNet-CIFAR10
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os
from datetime import datetime

def extract_tensor_values(series):
    """Extract numeric values from tensor strings"""
    def extract_value(x):
        if isinstance(x, str) and 'tensor(' in x:
            # Extract number from tensor(value) format
            match = re.search(r'tensor\(([0-9.]+)\)', x)
            if match:
                return float(match.group(1))
        elif isinstance(x, (int, float)):
            return float(x)
        elif isinstance(x, str):
            try:
                return float(x)
            except:
                return np.nan
        return np.nan
    
    return series.apply(extract_value)

def create_paper_style_comparison(train_stats_path, llc_results_path, output_dir):
    """Create comparison plots in paper style"""
    
    # Load training statistics
    print(f"Loading training statistics from: {train_stats_path}")
    train_stats = pd.read_csv(train_stats_path)
    
    # Load LLC results
    print(f"Loading LLC results from: {llc_results_path}")
    llc_stats = pd.read_csv(llc_results_path)
    
    # Clean training data - extract values from tensor strings
    print("Processing training data...")
    train_stats['train_acc_clean'] = extract_tensor_values(train_stats['train_acc']) * 100
    train_stats['test_acc_clean'] = extract_tensor_values(train_stats['test_acc']) * 100
    train_stats['adv_acc_clean'] = extract_tensor_values(train_stats['adv_acc']) * 100
    
    # Create figure with 2x2 subplots - match paper dimensions and styling
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Set overall figure styling to match paper
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100
    })
    
    # Plot 1: Accuracy curves (top left) - Show ONLY actual measured data
    print("Creating accuracy plots...")
    
    # Clean Test Accuracy
    ax1.semilogx(train_stats['step'], train_stats['test_acc_clean'], 
                 '-', color='#8B4513', linewidth=2, label='Clean Test')
    
    # Adversarial accuracy - ACTUAL measured data only (ε = 50/255 ≈ 0.196 from config)
    ax1.semilogx(train_stats['step'], train_stats['adv_acc_clean'], 
                 '-', color='#FF4500', linewidth=2, label='Adv. ε = 0.196')
    
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 80)  # Match paper's y-axis range
    ax1.set_xlim(1e0, 5e5)  # Extended to capture full effect at 500K steps
    
    # Plot 2: Local Complexity (top right) - Match paper exactly
    print("Creating local complexity plot...")
    
    # Match paper's exact colors and styling
    ax2.semilogx(train_stats['step'], train_stats['train_LC'], 
                 '-', color='green', linewidth=2, label='train')
    ax2.semilogx(train_stats['step'], train_stats['test_LC'], 
                 '-', color='brown', linewidth=2, label='test')  
    ax2.semilogx(train_stats['step'], train_stats['rand_LC'], 
                 '-', color='blue', linewidth=2, label='random')
    
    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('Local Complexity')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e0, 5e5)  # Extended to capture full effect at 500K steps
    ax2.set_ylim(20, 45)  # Set specific range for Local Complexity as requested
    
    # Plot 3: LLC Evolution (bottom left) - Match paper styling
    print("Creating LLC evolution plot...")
    
    # Filter successful LLC results
    llc_success = llc_stats[llc_stats['success'] == True].copy()
    
    ax3.semilogx(llc_success['step'], llc_success['llc_mean'], 
                 '-', color='purple', linewidth=2, label='LLC')
    
    # Add error bars if std is available
    if 'llc_std' in llc_success.columns and not llc_success['llc_std'].isna().all():
        valid_std = llc_success.dropna(subset=['llc_std'])
        if len(valid_std) > 0:
            ax3.fill_between(valid_std['step'], 
                           valid_std['llc_mean'] - valid_std['llc_std'],
                           valid_std['llc_mean'] + valid_std['llc_std'],
                           alpha=0.3, color='purple')
    
    ax3.set_xlabel('Optimization Steps')
    ax3.set_ylabel('Mean Local Learning Coefficient')
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1e0, 5e5)  # Extended to capture full effect at 500K steps
    
    # Rescale y-axis to make small LLC values visible
    if len(llc_success) > 0:
        llc_min = llc_success['llc_mean'].min()
        llc_max = llc_success['llc_mean'].max()
        llc_range = llc_max - llc_min
        
        # Add 10% padding above and below
        padding = llc_range * 0.1 if llc_range > 0 else abs(llc_min) * 0.1
        y_min = llc_min - padding
        y_max = llc_max + padding
        
        # Ensure we don't set a range that's too small
        if llc_range < 1e-6:
            y_center = (llc_min + llc_max) / 2
            y_min = y_center - 1e-6
            y_max = y_center + 1e-6
        
        ax3.set_ylim(y_min, y_max)
        print(f"LLC y-axis rescaled to [{y_min:.6f}, {y_max:.6f}] (range: {llc_range:.6f})")
    else:
        print("No LLC data found for y-axis scaling")
    
    # Plot 4: Combined comparison (bottom right) - Match paper styling
    print("Creating combined comparison...")
    
    # Normalize both metrics to [0, 1] for comparison
    lc_train_norm = (train_stats['train_LC'] - train_stats['train_LC'].min()) / \
                    (train_stats['train_LC'].max() - train_stats['train_LC'].min())
    lc_test_norm = (train_stats['test_LC'] - train_stats['test_LC'].min()) / \
                   (train_stats['test_LC'].max() - train_stats['test_LC'].min())
    
    if len(llc_success) > 0:
        llc_norm = (llc_success['llc_mean'] - llc_success['llc_mean'].min()) / \
                   (llc_success['llc_mean'].max() - llc_success['llc_mean'].min())
    
    ax4.semilogx(train_stats['step'], lc_train_norm, 
                 '-', color='green', linewidth=2, label='LC (train)', alpha=0.8)
    ax4.semilogx(train_stats['step'], lc_test_norm, 
                 '-', color='brown', linewidth=2, label='LC (test)', alpha=0.8)
    
    if len(llc_success) > 0:
        ax4.semilogx(llc_success['step'], llc_norm, 
                     '-', color='purple', linewidth=2, label='LLC (normalized)')
    
    ax4.set_xlabel('Optimization Steps')
    ax4.set_ylabel('Normalized Complexity')
    ax4.legend(loc='upper right', frameon=True, fancybox=True, shadow=False)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1e0, 5e5)  # Extended to capture full effect at 500K steps
    ax4.set_ylim(0, 1)  # Normalized range
    
    plt.tight_layout()
    
    # Save the plot with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'paper_style_comparison_{timestamp}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Paper-style comparison plot saved to: {output_path}")
    
    # Print some statistics
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    
    print(f"Training data points: {len(train_stats)}")
    print(f"LLC data points: {len(llc_success)}")
    
    if len(train_stats) > 0:
        print(f"\nAccuracy ranges:")
        print(f"  Clean test: {train_stats['test_acc_clean'].min():.1f}% - {train_stats['test_acc_clean'].max():.1f}%")
        print(f"  Adversarial: {train_stats['adv_acc_clean'].min():.1f}% - {train_stats['adv_acc_clean'].max():.1f}%")
        
        print(f"\nLocal Complexity ranges:")
        print(f"  Train LC: {train_stats['train_LC'].min():.1f} - {train_stats['train_LC'].max():.1f}")
        print(f"  Test LC: {train_stats['test_LC'].min():.1f} - {train_stats['test_LC'].max():.1f}")
        print(f"  Random LC: {train_stats['rand_LC'].min():.1f} - {train_stats['rand_LC'].max():.1f}")
    
    if len(llc_success) > 0:
        print(f"\nLLC range: {llc_success['llc_mean'].min():.3f} - {llc_success['llc_mean'].max():.3f}")
    
    return fig

def create_detailed_analysis(train_stats_path, llc_results_path, output_dir):
    """Create additional detailed analysis plots"""
    
    # Load data
    print(f"Loading training statistics from: {train_stats_path}")
    train_stats = pd.read_csv(train_stats_path)
    print(f"Loading LLC results from: {llc_results_path}")
    llc_stats = pd.read_csv(llc_results_path)
    
    # Process data
    train_stats['test_acc_clean'] = extract_tensor_values(train_stats['test_acc']) * 100
    train_stats['adv_acc_clean'] = extract_tensor_values(train_stats['adv_acc']) * 100
    llc_success = llc_stats[llc_stats['success'] == True].copy()
    
    # Create figure for detailed analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Grokking phases identification
    grok_threshold = 70  # Define grokking as >70% test accuracy
    grok_mask = train_stats['test_acc_clean'] > grok_threshold
    grok_start = train_stats[grok_mask]['step'].min() if grok_mask.any() else None
    
    ax1.semilogx(train_stats['step'], train_stats['test_acc_clean'], 'o-', color='blue', label='Test Accuracy')
    ax1.semilogx(train_stats['step'], train_stats['adv_acc_clean'], 'o-', color='red', label='Adversarial Accuracy')
    
    if grok_start:
        ax1.axvline(grok_start, color='green', linestyle='--', alpha=0.7, label=f'Grokking starts (~{grok_start})')
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Grokking Phase Identification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LC vs LLC correlation analysis
    if len(llc_success) > 0:
        # Interpolate LLC values to match training steps
        llc_interp = np.interp(train_stats['step'], llc_success['step'], llc_success['llc_mean'])
        
        # Create scatter plot
        ax2.scatter(train_stats['train_LC'], llc_interp, alpha=0.6, c=train_stats['step'], 
                   cmap='viridis', s=30)
        ax2.set_xlabel('Local Complexity (Train)')
        ax2.set_ylabel('LLC Mean')
        ax2.set_title('LC vs LLC Correlation')
        
        # Add correlation coefficient
        valid_mask = ~np.isnan(llc_interp)
        if valid_mask.sum() > 1:
            corr = np.corrcoef(train_stats['train_LC'][valid_mask], llc_interp[valid_mask])[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.colorbar(ax2.collections[0], ax=ax2, label='Training Step')
    
    # Plot 3: Complexity evolution phases
    ax3.semilogx(train_stats['step'], train_stats['train_LC'], label='Train LC', linewidth=2)
    ax3.semilogx(train_stats['step'], train_stats['test_LC'], label='Test LC', linewidth=2)
    ax3.semilogx(train_stats['step'], train_stats['rand_LC'], label='Random LC', linewidth=2)
    
    if len(llc_success) > 0:
        # Scale LLC to be visible with LC
        llc_scaled = llc_success['llc_mean'] * 10 + 30  # Rough scaling for visibility
        ax3.semilogx(llc_success['step'], llc_scaled, 'o-', label='LLC (scaled)', 
                    linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Complexity Measure')
    ax3.set_title('All Complexity Measures Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Robustness vs Complexity
    ax4.scatter(train_stats['train_LC'], train_stats['adv_acc_clean'], 
               alpha=0.6, c=train_stats['step'], cmap='plasma', s=40)
    ax4.set_xlabel('Local Complexity (Train)')
    ax4.set_ylabel('Adversarial Accuracy (%)')
    ax4.set_title('Complexity vs Robustness')
    plt.colorbar(ax4.collections[0], ax=ax4, label='Training Step')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save detailed analysis with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'detailed_analysis_{timestamp}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed analysis saved to: {output_path}")
    
    return fig

def create_clean_vs_adversarial_comparison(train_stats_path, llc_clean_path, llc_adversarial_path, output_dir):
    """Create comparison between clean and adversarial trajectories"""
    
    # Load data
    print(f"Loading training statistics from: {train_stats_path}")
    train_stats = pd.read_csv(train_stats_path)
    
    print(f"Loading clean LLC results from: {llc_clean_path}")
    llc_clean = pd.read_csv(llc_clean_path)
    
    print(f"Loading adversarial LLC results from: {llc_adversarial_path}")
    llc_adversarial = pd.read_csv(llc_adversarial_path)
    
    # Process data
    train_stats['test_acc_clean'] = extract_tensor_values(train_stats['test_acc']) * 100
    train_stats['adv_acc_clean'] = extract_tensor_values(train_stats['adv_acc']) * 100
    
    # Filter successful LLC results
    llc_clean_success = llc_clean[llc_clean['success'] == True].copy()
    llc_adversarial_success = llc_adversarial[llc_adversarial['success'] == True].copy()
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Set styling
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100
    })
    
    # Plot 1: Test Accuracy Comparison (top left)
    ax1.semilogx(train_stats['step'], train_stats['test_acc_clean'], 
                 '-', color='#2E8B57', linewidth=2, label='Clean Test Acc')
    ax1.semilogx(train_stats['step'], train_stats['adv_acc_clean'], 
                 '-', color='#DC143C', linewidth=2, label='Adversarial Test Acc')
    
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Clean vs Adversarial Test Accuracy')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 80)
    ax1.set_xlim(1e0, 5e5)
    
    # Plot 2: LLC Comparison (top right)
    if len(llc_clean_success) > 0:
        ax2.semilogx(llc_clean_success['step'], llc_clean_success['llc_mean'], 
                     'o-', color='#2E8B57', linewidth=2, markersize=4, 
                     label='LLC on Clean Data', alpha=0.8)
        
        # Add error bars for clean LLC
        if 'llc_std' in llc_clean_success.columns and not llc_clean_success['llc_std'].isna().all():
            valid_clean = llc_clean_success.dropna(subset=['llc_std'])
            if len(valid_clean) > 0:
                ax2.fill_between(valid_clean['step'], 
                               valid_clean['llc_mean'] - valid_clean['llc_std'],
                               valid_clean['llc_mean'] + valid_clean['llc_std'],
                               alpha=0.2, color='#2E8B57')
    
    if len(llc_adversarial_success) > 0:
        ax2.semilogx(llc_adversarial_success['step'], llc_adversarial_success['llc_mean'], 
                     's-', color='#DC143C', linewidth=2, markersize=4, 
                     label='LLC on Adversarial Data', alpha=0.8)
        
        # Add error bars for adversarial LLC
        if 'llc_std' in llc_adversarial_success.columns and not llc_adversarial_success['llc_std'].isna().all():
            valid_adv = llc_adversarial_success.dropna(subset=['llc_std'])
            if len(valid_adv) > 0:
                ax2.fill_between(valid_adv['step'], 
                               valid_adv['llc_mean'] - valid_adv['llc_std'],
                               valid_adv['llc_mean'] + valid_adv['llc_std'],
                               alpha=0.2, color='#DC143C')
    
    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('Local Learning Coefficient')
    ax2.set_title('LLC: Clean vs Adversarial Data')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e0, 5e5)
    
    # Plot 3: Combined Normalized Comparison (bottom left)
    # Normalize accuracies to [0, 1] for comparison
    test_acc_norm = train_stats['test_acc_clean'] / 100.0
    adv_acc_norm = train_stats['adv_acc_clean'] / 100.0
    
    ax3.semilogx(train_stats['step'], test_acc_norm, 
                 '-', color='#2E8B57', linewidth=2, label='Clean Test Acc (norm)')
    ax3.semilogx(train_stats['step'], adv_acc_norm, 
                 '-', color='#DC143C', linewidth=2, label='Adv Test Acc (norm)')
    
    # Normalize LLC values if available
    if len(llc_clean_success) > 0:
        llc_clean_norm = (llc_clean_success['llc_mean'] - llc_clean_success['llc_mean'].min()) / \
                        (llc_clean_success['llc_mean'].max() - llc_clean_success['llc_mean'].min())
        ax3.semilogx(llc_clean_success['step'], llc_clean_norm, 
                     'o-', color='#228B22', linewidth=2, markersize=3, 
                     label='LLC Clean (norm)', alpha=0.7)
    
    if len(llc_adversarial_success) > 0:
        llc_adv_norm = (llc_adversarial_success['llc_mean'] - llc_adversarial_success['llc_mean'].min()) / \
                      (llc_adversarial_success['llc_mean'].max() - llc_adversarial_success['llc_mean'].min())
        ax3.semilogx(llc_adversarial_success['step'], llc_adv_norm, 
                     's-', color='#B22222', linewidth=2, markersize=3, 
                     label='LLC Adv (norm)', alpha=0.7)
    
    ax3.set_xlabel('Optimization Steps')
    ax3.set_ylabel('Normalized Values')
    ax3.set_title('Normalized Comparison: Accuracy vs LLC')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1e0, 5e5)
    ax3.set_ylim(0, 1)
    
    # Plot 4: LLC Difference Analysis (bottom right)
    if len(llc_clean_success) > 0 and len(llc_adversarial_success) > 0:
        # Interpolate to common steps for difference calculation
        common_steps = np.intersect1d(llc_clean_success['step'], llc_adversarial_success['step'])
        if len(common_steps) > 0:
            clean_interp = llc_clean_success.set_index('step').loc[common_steps, 'llc_mean']
            adv_interp = llc_adversarial_success.set_index('step').loc[common_steps, 'llc_mean']
            
            llc_difference = adv_interp.values - clean_interp.values
            
            ax4.semilogx(common_steps, llc_difference, 
                        'o-', color='#4B0082', linewidth=2, markersize=4, 
                        label='LLC(Adv) - LLC(Clean)')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax4.set_xlabel('Optimization Steps')
            ax4.set_ylabel('LLC Difference')
            ax4.set_title('LLC Difference: Adversarial - Clean')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(1e0, 5e5)
        else:
            ax4.text(0.5, 0.5, 'No overlapping steps\nbetween clean and\nadversarial LLC data', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('LLC Difference: Adversarial - Clean')
    else:
        ax4.text(0.5, 0.5, 'Insufficient LLC data\nfor difference analysis', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('LLC Difference: Adversarial - Clean')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'clean_vs_adversarial_comparison_{timestamp}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Clean vs Adversarial comparison plot saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("CLEAN VS ADVERSARIAL COMPARISON STATISTICS")
    print("="*60)
    
    print(f"Training data points: {len(train_stats)}")
    print(f"Clean LLC data points: {len(llc_clean_success)}")
    print(f"Adversarial LLC data points: {len(llc_adversarial_success)}")
    
    if len(train_stats) > 0:
        print(f"\nTest Accuracy ranges:")
        print(f"  Clean: {train_stats['test_acc_clean'].min():.1f}% - {train_stats['test_acc_clean'].max():.1f}%")
        print(f"  Adversarial: {train_stats['adv_acc_clean'].min():.1f}% - {train_stats['adv_acc_clean'].max():.1f}%")
    
    if len(llc_clean_success) > 0:
        print(f"\nClean LLC range: {llc_clean_success['llc_mean'].min():.3f} - {llc_clean_success['llc_mean'].max():.3f}")
    
    if len(llc_adversarial_success) > 0:
        print(f"Adversarial LLC range: {llc_adversarial_success['llc_mean'].min():.3f} - {llc_adversarial_success['llc_mean'].max():.3f}")
    
    return fig

def create_loss_llc_comparison(train_stats_path, llc_results_path, output_dir, zoomed_in=False, log_scale=False):
    """Create comparison between train/test loss and LLC trajectory"""
    
    # Load data
    print(f"Loading training statistics from: {train_stats_path}")
    train_stats = pd.read_csv(train_stats_path)
    
    print(f"Loading LLC results from: {llc_results_path}")
    llc_stats = pd.read_csv(llc_results_path)
    
    # Detect if this is adversarial results based on path (excluding base directory name)
    import os
    # Remove the base directory name to avoid false positives from "grok-adversarial"
    path_parts = llc_results_path.split(os.sep)
    # Look for "adversarial" in directory names or filename, but skip the first few parts that might contain "grok-adversarial"
    relevant_path = os.sep.join(path_parts[-3:]).lower()  # Last 3 parts of path
    is_adversarial = "adversarial" in relevant_path and "grok-adversarial" not in relevant_path
    print(f"Detected {'adversarial' if is_adversarial else 'clean'} LLC results based on path: {relevant_path}")
    
    # Process data
    train_stats['train_loss_clean'] = extract_tensor_values(train_stats['train_loss'])
    train_stats['test_loss_clean'] = extract_tensor_values(train_stats['test_loss'])
    train_stats['test_acc_clean'] = extract_tensor_values(train_stats['test_acc']) * 100
    
    # Debug: Print data ranges
    print(f"Train loss range: {train_stats['train_loss_clean'].min():.4f} - {train_stats['train_loss_clean'].max():.4f}")
    print(f"Test loss range: {train_stats['test_loss_clean'].min():.4f} - {train_stats['test_loss_clean'].max():.4f}")
    print(f"Test accuracy range: {train_stats['test_acc_clean'].min():.2f}% - {train_stats['test_acc_clean'].max():.2f}%")
    
    # Process adversarial accuracy if available
    if is_adversarial and 'adv_acc' in train_stats.columns:
        train_stats['adv_acc_clean'] = extract_tensor_values(train_stats['adv_acc']) * 100
        print("Found adversarial accuracy data - will plot adversarial robustness instead of test loss")
    
    # Filter successful LLC results
    llc_success = llc_stats[llc_stats['success'] == True].copy()
    
    # Debug: Print LLC data info
    print(f"Total LLC entries: {len(llc_stats)}")
    print(f"Successful LLC entries: {len(llc_success)}")
    if len(llc_success) > 0:
        print(f"LLC step range: {llc_success['step'].min()} - {llc_success['step'].max()}")
        print(f"LLC mean range: {llc_success['llc_mean'].min():.4f} - {llc_success['llc_mean'].max():.4f}")
    
    # Use grokking definition from original paper: starts around 10^4 optimization steps
    grok_start = 10000  # 10^4 as defined in the original grokking paper
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Set styling with larger fonts
    plt.rcParams.update({
        'font.size': 14,
        'axes.linewidth': 1,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })
    
    # Plot 1: Loss/Accuracy Evolution (top left)
    # Always use semilogx for x-axis, y-axis scaling is handled separately
    ax1.semilogx(train_stats['step'], train_stats['train_loss_clean'], 
                 '-', color='#1f77b4', linewidth=2, label='Train Loss')
    
    if is_adversarial and 'adv_acc_clean' in train_stats.columns:
        # Create dual y-axis for adversarial accuracy
        ax1_twin = ax1.twinx()
        if log_scale:
            # For adversarial accuracy, we can't use log scale since it includes 0-100%
            # So we use semilogx (log x-axis only)
            ax1_twin.semilogx(train_stats['step'], train_stats['adv_acc_clean'], 
                             '-', color='#ff7f0e', linewidth=2, label='Adversarial Accuracy')
        else:
            ax1_twin.semilogx(train_stats['step'], train_stats['adv_acc_clean'], 
                             '-', color='#ff7f0e', linewidth=2, label='Adversarial Accuracy')
        ax1.set_ylabel('Train Loss', color='#1f77b4', fontsize=15)
        ax1_twin.set_ylabel('Adversarial Accuracy (%)', color='#ff7f0e', fontsize=15)
        ax1.set_title('Train Loss vs Adversarial Robustness' + (' (Symlog Scale)' if log_scale else ''), fontsize=14)
        
        # Set y-axis limits
        if log_scale:
            # Use symlog scale for hybrid behavior: linear near zero, log for large values
            ax1.set_yscale('symlog', linthresh=1.0)
            max_loss = train_stats['train_loss_clean'].max()
            ax1.set_ylim(-0.2, max_loss * 1.1)
        else:
            ax1.set_ylim(0, train_stats['train_loss_clean'].max() * 1.1)
        ax1_twin.set_ylim(0, 100)
        
        # Don't create legend yet - wait for grokking indicator
        pass
    else:
        # Standard test loss plot
        ax1.semilogx(train_stats['step'], train_stats['test_loss_clean'], 
                     '-', color='#ff7f0e', linewidth=2, label='Test Loss')
        ax1.set_ylabel('Loss', fontsize=15)
        ax1.set_title('Train vs Test Loss Evolution' + (' (Symlog Scale)' if log_scale else ''), fontsize=14)
        # Don't create legend yet - wait for grokking indicator
        # Set y-axis to show the actual range of loss values
        if log_scale:
            # Use symlog scale for hybrid behavior: linear near zero, log for large values
            max_loss = max(train_stats['train_loss_clean'].max(), train_stats['test_loss_clean'].max())
            # Use reasonable linthresh for better scaling
            linthresh = 1.0  # Linear behavior for values 0-1, then logarithmic above
            ax1.set_yscale('symlog', linthresh=linthresh)  # Remove linscale parameter
            # Set limits to ensure the peak is fully visible with significant headroom
            upper_limit = max_loss * 1.5  # Give 50% more space above the peak
            ax1.set_ylim(0, upper_limit)  # Start from 0
            print(f"Setting symlog scale with linthresh={linthresh:.3f}, y-axis limit: 0 - {upper_limit:.4f} (max_loss: {max_loss:.4f})")
        elif zoomed_in:
            # Zoomed in view: focus on 0-10 range for better detail
            ax1.set_ylim(-0.2, 10)
            print(f"Setting zoomed y-axis limit to: -0.2 - 10")
        else:
            # Normal view: use full data range
            max_loss = max(train_stats['train_loss_clean'].max(), train_stats['test_loss_clean'].max())
            min_y = -max_loss * 0.02  # Show 2% of max loss below zero for better visibility
            ax1.set_ylim(min_y, max_loss * 1.1)
            print(f"Setting y-axis limit to: {min_y:.4f} - {max_loss * 1.1:.4f}")
    
    ax1.set_xlabel('Optimization Steps', fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e0, 5e5)
    
    # Add grokking indicator and create final legend
    if grok_start:
        ax1.axvline(grok_start, color='green', linestyle='--', alpha=0.7, 
                   label=f'Grokking starts (~{grok_start})')
    
    # Create legend after all elements are added
    if is_adversarial and 'adv_acc_clean' in train_stats.columns:
        # For adversarial case, combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')
    else:
        # For standard case, just use ax1 legend
        ax1.legend(loc='upper left')
    
    # Plot 2: LLC Evolution (top right)
    if len(llc_success) > 0:
        ax2.semilogx(llc_success['step'], llc_success['llc_mean'], 
                     'o-', color='#2ca02c', linewidth=2, markersize=4, 
                     label='LLC', alpha=0.8)
        
        # Add error bars if std is available
        if 'llc_std' in llc_success.columns and not llc_success['llc_std'].isna().all():
            valid_std = llc_success.dropna(subset=['llc_std'])
            if len(valid_std) > 0:
                ax2.fill_between(valid_std['step'], 
                               valid_std['llc_mean'] - valid_std['llc_std'],
                               valid_std['llc_mean'] + valid_std['llc_std'],
                               alpha=0.2, color='#2ca02c')
    
    ax2.set_xlabel('Optimization Steps', fontsize=15)
    ax2.set_ylabel('Local Learning Coefficient', fontsize=15)
    ax2.set_title('LLC Evolution', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e0, 5e5)
    
    # Add grokking indicator
    if grok_start:
        ax2.axvline(grok_start, color='green', linestyle='--', alpha=0.7, 
                   label=f'Grokking starts (~{grok_start})')
        ax2.legend(loc='upper left')
    
    # Plot 3: Combined Loss and LLC (bottom left)
    # Use dual y-axis to show both loss and LLC
    ax3_twin = ax3.twinx()
    
    # Plot train loss and test loss/adversarial accuracy on left y-axis
    line1 = ax3.semilogx(train_stats['step'], train_stats['train_loss_clean'], 
                        '-', color='#1f77b4', linewidth=2, label='Train Loss')
    
    if is_adversarial and 'adv_acc_clean' in train_stats.columns:
        # For adversarial results, plot adversarial accuracy instead of test loss
        # Adversarial accuracy can't use log y-axis since it includes 0-100%
        line2 = ax3.semilogx(train_stats['step'], train_stats['adv_acc_clean'], 
                            '-', color='#ff7f0e', linewidth=2, label='Adversarial Accuracy')
        ax3.set_ylabel('Train Loss / Adversarial Accuracy (%)', color='black', fontsize=15)
        title_suffix = ' (Symlog Scale)' if log_scale else ''
        ax3.set_title('Train Loss, Adversarial Accuracy and LLC' + title_suffix, fontsize=14)
        # Set y-axis to accommodate both loss and accuracy (0-100%)
        if log_scale:
            # Use symlog scale for hybrid behavior in combined plot
            max_loss = train_stats['train_loss_clean'].max()
            # Use reasonable linthresh for better scaling
            linthresh = 1.0  # Linear behavior for values 0-1, then logarithmic above
            ax3.set_yscale('symlog', linthresh=linthresh)  # Remove linscale parameter
            # Set limits to ensure the peak is fully visible with significant headroom
            upper_limit = max(max_loss * 1.5, 100)  # Give 50% more space above the peak, at least 100 for accuracy
            ax3.set_ylim(0, upper_limit)  # Start from 0
            print(f"Setting combined plot symlog scale with linthresh={linthresh:.3f}, y-axis limit: 0 - {upper_limit:.4f} (adversarial mode, max_loss: {max_loss:.4f})")
        elif zoomed_in:
            ax3.set_ylim(0, 10)
            print(f"Setting combined plot zoomed y-axis limit to: 0 - 10 (adversarial mode)")
        else:
            max_loss = train_stats['train_loss_clean'].max() * 1.1
            ax3.set_ylim(0, max(max_loss, 100))
    else:
        # Standard test loss plot
        line2 = ax3.semilogx(train_stats['step'], train_stats['test_loss_clean'], 
                            '-', color='#ff7f0e', linewidth=2, label='Test Loss')
        ax3.set_ylabel('Loss', color='black', fontsize=15)
        title_suffix = ' (Symlog Scale)' if log_scale else ''
        ax3.set_title('Loss and LLC Combined' + title_suffix, fontsize=14)
        # Set y-axis to show the actual range of loss values
        if log_scale:
            # Use symlog scale for hybrid behavior in combined plot
            max_loss = max(train_stats['train_loss_clean'].max(), train_stats['test_loss_clean'].max())
            # Use reasonable linthresh for better scaling
            linthresh = 1.0  # Linear behavior for values 0-1, then logarithmic above
            ax3.set_yscale('symlog', linthresh=linthresh)  # Remove linscale parameter
            # Set limits to ensure the peak is fully visible with significant headroom
            upper_limit = max_loss * 1.5  # Give 50% more space above the peak
            ax3.set_ylim(0, upper_limit)  # Start from 0
            print(f"Setting combined plot symlog scale with linthresh={linthresh:.3f}, y-axis limit: 0 - {upper_limit:.4f} (max_loss: {max_loss:.4f})")
        elif zoomed_in:
            ax3.set_ylim(0, 10)
            print(f"Setting combined plot zoomed y-axis limit to: 0 - 10")
        else:
            max_loss = max(train_stats['train_loss_clean'].max(), train_stats['test_loss_clean'].max())
            ax3.set_ylim(0, max_loss * 1.1)
            print(f"Setting combined plot y-axis limit to: 0 - {max_loss * 1.1:.4f}")
    
    # Plot LLC on right y-axis
    if len(llc_success) > 0:
        line3 = ax3_twin.semilogx(llc_success['step'], llc_success['llc_mean'], 
                                 'o-', color='#2ca02c', linewidth=2, markersize=3, 
                                 label='LLC', alpha=0.8)
    
    ax3.set_xlabel('Optimization Steps', fontsize=15)
    ax3_twin.set_ylabel('Local Learning Coefficient', color='#2ca02c', fontsize=15)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1e0, 5e5)  # Cut off x-axis at 10^4
    
    # Set LLC y-axis to align zero points horizontally with the loss axis
    if len(llc_success) > 0:
        llc_min = llc_success['llc_mean'].min()
        llc_max = llc_success['llc_mean'].max()
        llc_range = llc_max - llc_min
        padding = llc_range * 0.1 if llc_range > 0 else abs(llc_max) * 0.1
        
        # Set LLC y-axis limits with optional zoomed view
        if zoomed_in:
            # Zoomed view: focus on 0-10 range for LLC as well
            # Use a small tolerance to treat near-zero values as zero
            tolerance = 0.1  # Values within 0.1 of zero are considered zero
            if llc_min >= -tolerance:
                # LLC data is essentially all positive, start from 0 to align with loss axis
                print(f"LLC data is essentially positive: min={llc_min:.3f}, max={llc_max:.3f} (tolerance={tolerance})")
                ax3_twin.set_ylim(0, 10)
                print(f"Set LLC y-axis to (0, 10), actual limits: {ax3_twin.get_ylim()}")
                # Force the limits to stick
                ax3_twin.set_ylim(0, 10)
                print(f"After forcing again, actual limits: {ax3_twin.get_ylim()}")
            else:
                # If LLC has significant negative values, include the negative range
                print(f"LLC data has significant negative values: min={llc_min:.3f}, max={llc_max:.3f}")
                ax3_twin.set_ylim(min(llc_min - padding, -2), 10)
                print(f"Setting LLC zoomed y-axis limits to: {ax3_twin.get_ylim()}")
        else:
            # Normal view: align zero points horizontally
            if llc_min >= 0:
                # LLC data is all positive, start from 0 to align with loss axis
                ax3_twin.set_ylim(0, llc_max + padding)
            else:
                # LLC has negative values, but we still want to align the zero points
                ax3_twin.set_ylim(llc_min - padding, max(0, llc_max + padding))
            print(f"Setting LLC y-axis limits to: {ax3_twin.get_ylim()}")
    else:
        ax3_twin.set_ylim(0, 30)  # Fallback if no LLC data
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if len(llc_success) > 0:
        lines += line3
        labels += [line3[0].get_label()]
    
    # Add grokking indicator
    if grok_start:
        ax3.axvline(grok_start, color='green', linestyle='--', alpha=0.7, 
                   label=f'Grokking starts (~{grok_start})')
        labels.append(f'Grokking starts (~{grok_start})')
    
    # Position legend based on whether it's adversarial data
    legend_loc = 'center left' if is_adversarial and 'adv_acc_clean' in train_stats.columns else 'upper left'
    ax3.legend(lines, labels, loc=legend_loc)
    
    # Final enforcement of axis limits (in case matplotlib adjusted them)
    if zoomed_in and len(llc_success) > 0:
        llc_min = llc_success['llc_mean'].min()
        tolerance = 0.1  # Same tolerance as above
        if llc_min >= -tolerance:
            print(f"Final enforcement: Setting LLC y-axis to (0, 10) for essentially positive data (min={llc_min:.3f})")
            ax3_twin.set_ylim(0, 10)
            print(f"Final LLC y-axis limits: {ax3_twin.get_ylim()}")
    
    # Plot 4: Loss/Accuracy vs LLC Correlation (bottom right)
    if len(llc_success) > 0:
        # Interpolate LLC values to match training steps
        llc_interp = np.interp(train_stats['step'], llc_success['step'], llc_success['llc_mean'])
        
        if is_adversarial and 'adv_acc_clean' in train_stats.columns:
            # Create scatter plot with adversarial accuracy
            scatter = ax4.scatter(train_stats['adv_acc_clean'], llc_interp, 
                                alpha=0.6, c=train_stats['step'], cmap='viridis', s=30)
            ax4.set_xlabel('Adversarial Accuracy (%)')
            ax4.set_ylabel('LLC Mean')
            ax4.set_title('Adversarial Accuracy vs LLC Correlation')
            
            # Add correlation coefficient
            valid_mask = ~np.isnan(llc_interp)
            if valid_mask.sum() > 1:
                corr = np.corrcoef(train_stats['adv_acc_clean'][valid_mask], llc_interp[valid_mask])[0, 1]
                ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes, 
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        else:
            # Create scatter plot with test loss
            scatter = ax4.scatter(train_stats['test_loss_clean'], llc_interp, 
                                alpha=0.6, c=train_stats['step'], cmap='viridis', s=30)
            ax4.set_xlabel('Test Loss', fontsize=15)
            ax4.set_ylabel('LLC Mean', fontsize=15)
            ax4.set_title('Test Loss vs LLC Correlation', fontsize=14)
            
            # Add correlation coefficient
            valid_mask = ~np.isnan(llc_interp)
            if valid_mask.sum() > 1:
                corr = np.corrcoef(train_stats['test_loss_clean'][valid_mask], llc_interp[valid_mask])[0, 1]
                ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax4.transAxes, 
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.colorbar(scatter, ax=ax4, label='Training Step')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No LLC data available\nfor correlation analysis', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Test Loss vs LLC Correlation')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if log_scale:
        suffix += "_symlog_scale"
    if zoomed_in:
        suffix += "_zoomed"
    output_path = os.path.join(output_dir, f'loss_llc_comparison{suffix}_{timestamp}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss vs LLC comparison plot saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("LOSS VS LLC COMPARISON STATISTICS")
    print("="*60)
    
    print(f"Training data points: {len(train_stats)}")
    print(f"LLC data points: {len(llc_success)}")
    
    if len(train_stats) > 0:
        print(f"\nLoss ranges:")
        print(f"  Train loss: {train_stats['train_loss_clean'].min():.4f} - {train_stats['train_loss_clean'].max():.4f}")
        print(f"  Test loss: {train_stats['test_loss_clean'].min():.4f} - {train_stats['test_loss_clean'].max():.4f}")
    
    if len(llc_success) > 0:
        print(f"\nLLC range: {llc_success['llc_mean'].min():.6f} - {llc_success['llc_mean'].max():.6f}")
    
    print(f"\nGrokking starts at step: {grok_start} (as defined in original paper: 10^4)")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Create paper-style comparison plots between Local Complexity and LLC metrics')
    parser.add_argument('--train_stats', required=True, 
                       help='Path to training statistics CSV file')
    parser.add_argument('--llc_results', 
                       help='Path to LLC results CSV file (for single LLC analysis)')
    parser.add_argument('--llc_clean', 
                       help='Path to clean LLC results CSV file (for clean vs adversarial comparison)')
    parser.add_argument('--llc_adversarial', 
                       help='Path to adversarial LLC results CSV file (for clean vs adversarial comparison)')
    parser.add_argument('--output_dir', default='./plots',
                       help='Output directory for plots (default: ./plots)')
    parser.add_argument('--paper_style_only', action='store_true',
                       help='Only create paper-style comparison (skip detailed analysis)')
    parser.add_argument('--detailed_only', action='store_true',
                       help='Only create detailed analysis (skip paper-style comparison)')
    parser.add_argument('--clean_vs_adversarial', action='store_true',
                       help='Create clean vs adversarial comparison (requires --llc_clean and --llc_adversarial)')
    parser.add_argument('--loss_llc_comparison', action='store_true',
                       help='Create loss vs LLC comparison (requires --llc_results)')
    parser.add_argument('--zoomed_in', action='store_true',
                       help='Zoom in on loss and LLC trajectories (0-10 range) for better detail visibility')
    parser.add_argument('--log_scale', action='store_true',
                       help='Use logarithmic y-axis scaling for loss plots to better visualize initial stage trends')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.train_stats):
        print(f"❌ Training stats file not found: {args.train_stats}")
        return
    
    # Validate based on analysis type
    if args.clean_vs_adversarial:
        # Clean vs adversarial comparison mode
        if not args.llc_clean or not args.llc_adversarial:
            print("❌ Clean vs adversarial comparison requires both --llc_clean and --llc_adversarial")
            return
        
        if not os.path.exists(args.llc_clean):
            print(f"❌ Clean LLC results file not found: {args.llc_clean}")
            return
            
        if not os.path.exists(args.llc_adversarial):
            print(f"❌ Adversarial LLC results file not found: {args.llc_adversarial}")
            return
        
        print("Creating clean vs adversarial comparison...")
        print("="*60)
        print(f"Training stats: {args.train_stats}")
        print(f"Clean LLC results: {args.llc_clean}")
        print(f"Adversarial LLC results: {args.llc_adversarial}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
        fig = create_clean_vs_adversarial_comparison(args.train_stats, args.llc_clean, args.llc_adversarial, args.output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files in:", args.output_dir)
        print("  - clean_vs_adversarial_comparison_[timestamp].png: Clean vs Adversarial comparison")
        print("  (Timestamp format: YYYYMMDD_HHMMSS)")
        print("="*60)
        
    elif args.loss_llc_comparison:
        # Loss vs LLC comparison mode
        if not args.llc_results:
            print("❌ Loss vs LLC comparison requires --llc_results")
            return
            
        if not os.path.exists(args.llc_results):
            print(f"❌ LLC results file not found: {args.llc_results}")
            return
        
        print("Creating loss vs LLC comparison...")
        print("="*60)
        print(f"Training stats: {args.train_stats}")
        print(f"LLC results: {args.llc_results}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
        fig = create_loss_llc_comparison(args.train_stats, args.llc_results, args.output_dir, zoomed_in=args.zoomed_in, log_scale=args.log_scale)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files in:", args.output_dir)
        print("  - loss_llc_comparison_[timestamp].png: Loss vs LLC comparison")
        print("  (Timestamp format: YYYYMMDD_HHMMSS)")
        print("="*60)
        
    else:
        # Standard single LLC analysis mode
        if not args.llc_results:
            print("❌ Standard analysis requires --llc_results")
            return
            
        if not os.path.exists(args.llc_results):
            print(f"❌ LLC results file not found: {args.llc_results}")
            return
        
        print("Creating comparison plots...")
        print("="*60)
        print(f"Training stats: {args.train_stats}")
        print(f"LLC results: {args.llc_results}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
        # Create plots based on arguments
        if not args.detailed_only:
            print("\nCreating paper-style comparison...")
            fig1 = create_paper_style_comparison(args.train_stats, args.llc_results, args.output_dir)
        
        if not args.paper_style_only:
            print("\nCreating detailed analysis...")
            fig2 = create_detailed_analysis(args.train_stats, args.llc_results, args.output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files in:", args.output_dir)
        if not args.detailed_only:
            print("  - paper_style_comparison_[timestamp].png: Main comparison in paper style")
        if not args.paper_style_only:
            print("  - detailed_analysis_[timestamp].png: Additional detailed analysis")
    print("  (Timestamp format: YYYYMMDD_HHMMSS)")
    print("="*60)

if __name__ == '__main__':
    main()
