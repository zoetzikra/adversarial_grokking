#!/usr/bin/env python3
"""
Script to analyze training statistics saved during training.
This allows you to access all the evaluation values outside of WandB
and create custom plots combining different metrics.
"""

import torch as ch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def load_training_stats(log_dir):
    """
    Load training statistics from the saved .pt file
    
    Args:
        log_dir: Path to the logs directory (e.g., './logs/Fri_Jul_18_17:04:54_2025')
    
    Returns:
        stats: Dictionary containing all training statistics
    """
    stats_path = os.path.join(log_dir, 'stats.pt')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    # Load with CPU mapping to handle CUDA tensors
    stats = ch.load(stats_path, map_location=ch.device('cpu'))
    print(f"Loaded stats from: {stats_path}")
    print(f"Available keys: {list(stats.keys())}")
    print(f"Number of data points: {len(stats['train_step'])}")
    
    return stats

def plot_training_curves(stats, save_path=None):
    """
    Create comprehensive training curves plot
    
    Args:
        stats: Training statistics dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Progress Analysis', fontsize=16)
    
    steps = np.array(stats['train_step'])
    
    # Plot 1: Accuracy
    axes[0, 0].plot(steps, stats['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(steps, stats['test_acc'], 'r-', label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[0, 1].plot(steps, stats['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(steps, stats['test_loss'], 'r-', label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Adversarial Accuracy
    if 'adv_acc' in stats:
        axes[0, 2].plot(steps, stats['adv_acc'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Training Step')
        axes[0, 2].set_ylabel('Adversarial Accuracy')
        axes[0, 2].set_title('Adversarial Robustness')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Local Complexity - Train
    if 'train_LC' in stats:
        train_lc = np.array([lc.sum(1).mean(0).item() for lc in stats['train_LC']])
        axes[1, 0].plot(steps, train_lc, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Local Complexity')
        axes[1, 0].set_title('Train Local Complexity')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Local Complexity - Test
    if 'test_LC' in stats:
        test_lc = np.array([lc.sum(1).mean(0).item() for lc in stats['test_LC']])
        axes[1, 1].plot(steps, test_lc, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Local Complexity')
        axes[1, 1].set_title('Test Local Complexity')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Local Complexity - Random
    if 'rand_LC' in stats:
        rand_lc = np.array([lc.sum(1).mean(0).item() for lc in stats['rand_LC']])
        axes[1, 2].plot(steps, rand_lc, 'purple', linewidth=2)
        axes[1, 2].set_xlabel('Training Step')
        axes[1, 2].set_ylabel('Local Complexity')
        axes[1, 2].set_title('Random Local Complexity')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def create_custom_comparison_plot(stats, your_complexity_metric=None, save_path=None):
    """
    Create a custom plot comparing LC measurements with your complexity metric
    
    Args:
        stats: Training statistics dictionary
        your_complexity_metric: Your post-training complexity measurements (optional)
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps = np.array(stats['train_step'])
    
    # Plot 1: Local Complexity measurements
    if 'train_LC' in stats:
        train_lc = np.array([lc.sum(1).mean(0).item() for lc in stats['train_LC']])
        ax1.plot(steps, train_lc, 'b-', label='Train LC', linewidth=2)
    
    if 'test_LC' in stats:
        test_lc = np.array([lc.sum(1).mean(0).item() for lc in stats['test_LC']])
        ax1.plot(steps, test_lc, 'r-', label='Test LC', linewidth=2)
    
    if 'rand_LC' in stats:
        rand_lc = np.array([lc.sum(1).mean(0).item() for lc in stats['rand_LC']])
        ax1.plot(steps, rand_lc, 'purple', label='Random LC', linewidth=2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Local Complexity')
    ax1.set_title('Local Complexity Measurements During Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Your complexity metric (if provided)
    if your_complexity_metric is not None:
        # Assuming your_complexity_metric is a list/array with same length as steps
        ax2.plot(steps, your_complexity_metric, 'g-', label='Your Complexity Metric', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Your Complexity Metric')
        ax2.set_title('Your Post-Training Complexity Measurements')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Custom comparison plot saved to: {save_path}")
    
    plt.show()

def export_stats_to_csv(stats, output_path):
    """
    Export training statistics to CSV for further analysis
    
    Args:
        stats: Training statistics dictionary
        output_path: Path to save the CSV file
    """
    import pandas as pd
    
    # Create a DataFrame
    data = {
        'step': stats['train_step'],
        'train_acc': stats['train_acc'],
        'test_acc': stats['test_acc'],
        'train_loss': stats['train_loss'],
        'test_loss': stats['test_loss'],
    }
    
    # Add LC measurements if available
    if 'train_LC' in stats:
        data['train_LC'] = [lc.sum(1).mean(0).item() for lc in stats['train_LC']]
    if 'test_LC' in stats:
        data['test_LC'] = [lc.sum(1).mean(0).item() for lc in stats['test_LC']]
    if 'rand_LC' in stats:
        data['rand_LC'] = [lc.sum(1).mean(0).item() for lc in stats['rand_LC']]
    if 'adv_acc' in stats:
        data['adv_acc'] = stats['adv_acc']
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Statistics exported to CSV: {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze training statistics')
    parser.add_argument('--log_dir', required=True, 
                       help='Path to logs directory (e.g., ./logs/Fri_Jul_18_17:04:54_2025)')
    parser.add_argument('--output_dir', default='./analysis_output',
                       help='Directory to save analysis outputs')
    parser.add_argument('--export_csv', action='store_true',
                       help='Export statistics to CSV')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load statistics
    stats = load_training_stats(args.log_dir)
    
    # Create plots
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(stats, save_path=plot_path)
    
    # Create custom comparison plot
    custom_plot_path = os.path.join(args.output_dir, 'custom_comparison.png')
    create_custom_comparison_plot(stats, save_path=custom_plot_path)
    
    # Export to CSV if requested
    if args.export_csv:
        csv_path = os.path.join(args.output_dir, 'training_stats.csv')
        export_stats_to_csv(stats, csv_path)
    
    print(f"\nAnalysis complete! Outputs saved to: {args.output_dir}")
    print(f"Available data points: {len(stats['train_step'])}")
    print(f"Training steps: {stats['train_step'][0]} to {stats['train_step'][-1]}")

if __name__ == '__main__':
    main() 