#!/usr/bin/env python3
"""
Create paper-style comparison plots between Local Complexity and LLC metrics
Reproduces the style from the "Deep Networks Always Grok" paper for ResNet-CIFAR10
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
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

def create_paper_style_comparison():
    """Create comparison plots in paper style"""
    
    # Load training statistics
    print("Loading training statistics...")
    train_stats = pd.read_csv('/home/ztzifa/grok-adversarial/analysis_output/training_stats.csv')
    
    # Load LLC results
    print("Loading LLC results...")
    llc_stats = pd.read_csv('/home/ztzifa/grok-adversarial/llc_full_calibration_results_fixed_gamma_1/retrospective_results/llc_nearest_neighbor_results.csv')
    
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
    output_path = f'/home/ztzifa/grok-adversarial/paper_style_comparison_{timestamp}.png'
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

def create_detailed_analysis():
    """Create additional detailed analysis plots"""
    
    # Load data
    train_stats = pd.read_csv('/home/ztzifa/grok-adversarial/analysis_output/training_stats.csv')
    llc_stats = pd.read_csv('/home/ztzifa/grok-adversarial/llc_full_calibration_results/retrospective_results/llc_nearest_neighbor_results.csv')
    
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
    output_path = f'/home/ztzifa/grok-adversarial/detailed_analysis_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed analysis saved to: {output_path}")
    
    return fig

if __name__ == '__main__':
    print("Creating paper-style comparison plots...")
    print("="*60)
    
    # Create main comparison
    fig1 = create_paper_style_comparison()
    
    print("\nCreating detailed analysis...")
    fig2 = create_detailed_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("  - paper_style_comparison_[timestamp].png: Main comparison in paper style")
    print("  - detailed_analysis_[timestamp].png: Additional detailed analysis")
    print("  (Timestamp format: YYYYMMDD_HHMMSS)")
    print("="*60)
