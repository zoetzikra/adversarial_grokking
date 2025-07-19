import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True,
                       help='Directory containing both training stats and LLC results')
    args = parser.parse_args()
    
    # Load original training results
    stats_file = os.path.join(args.results_dir, 'stats.pt')
    llc_file = os.path.join(args.results_dir, 'llc_retrospective_results.pt')
    
    if not os.path.exists(stats_file):
        print(f"Training stats not found at {stats_file}")
        return
        
    if not os.path.exists(llc_file):
        print(f"LLC results not found at {llc_file}")
        return
    
    # Load data
    training_stats = torch.load(stats_file)
    llc_results = torch.load(llc_file)
    
    # Create unified dataframe
    training_df = pd.DataFrame({
        'step': training_stats['train_step'],
        'train_acc': training_stats['train_acc'],
        'test_acc': training_stats['test_acc'],
        'train_loss': training_stats['train_loss'],
        'test_loss': training_stats['test_loss'],
        'adv_acc': training_stats['adv_acc'],
        'train_lc': [lc.sum(1).mean(0).item() for lc in training_stats['train_LC']],
        'test_lc': [lc.sum(1).mean(0).item() for lc in training_stats['test_LC']],
        'rand_lc': [lc.sum(1).mean(0).item() for lc in training_stats['rand_LC']]
    })
    
    llc_df = pd.DataFrame(llc_results)
    
    # Merge on step
    combined_df = pd.merge(training_df, llc_df, on='step', how='outer')
    combined_df = combined_df.sort_values('step')
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: LLC vs Local Complexity
    ax1 = axes[0, 0]
    ax1.plot(combined_df['step'], combined_df['llc_mean'], 'g-', label='LLC', linewidth=2)
    ax1.fill_between(combined_df['step'], 
                     combined_df['llc_mean'] - combined_df['llc_std'],
                     combined_df['llc_mean'] + combined_df['llc_std'],
                     alpha=0.3, color='green')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(combined_df['step'], combined_df['train_lc'], 'b--', label='Train LC')
    ax1_twin.plot(combined_df['step'], combined_df['test_lc'], 'r--', label='Test LC')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('LLC', color='green')
    ax1_twin.set_ylabel('Local Complexity (Linear Regions)', color='blue')
    ax1.set_title('LLC vs Local Complexity')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Accuracy Evolution
    ax2 = axes[0, 1]
    ax2.plot(combined_df['step'], combined_df['test_acc'], 'k-', label='Test Accuracy')
    ax2.plot(combined_df['step'], combined_df['adv_acc'], 'orange', label='Adversarial Accuracy')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Performance Evolution')
    ax2.legend()
    
    # Plot 3: Complexity Ratio
    ax3 = axes[1, 0]
    complexity_ratio = combined_df['llc_mean'] / combined_df['train_lc']
    ax3.plot(combined_df['step'], complexity_ratio, 'purple', label='LLC / Local Complexity')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Complexity Ratio')
    ax3.set_title('LLC vs Local Complexity Ratio')
    ax3.legend()
    
    # Plot 4: Loss Evolution
    ax4 = axes[1, 1]
    ax4.plot(combined_df['step'], combined_df['train_loss'], 'b-', label='Train Loss')
    ax4.plot(combined_df['step'], combined_df['test_loss'], 'r-', label='Test Loss')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Loss')
    ax4.set_title('Loss Evolution')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(args.results_dir, 'llc_analysis_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to {output_file}")
    
    # Save combined dataframe
    csv_output = os.path.join(args.results_dir, 'combined_analysis.csv')
    combined_df.to_csv(csv_output, index=False)
    print(f"Combined data saved to {csv_output}")
    
    # Print summary statistics
    print("\n=== GROKKING ANALYSIS SUMMARY ===")
    
    # Find approximate grokking point (when test acc > 0.6)
    grok_mask = combined_df['test_acc'] > 0.6
    if grok_mask.any():
        grok_step = combined_df[grok_mask]['step'].iloc[0]
        print(f"Approximate grokking onset: step {grok_step}")
        
        # Compare complexity before/after grokking
        pre_grok = combined_df[combined_df['step'] < grok_step]
        post_grok = combined_df[combined_df['step'] >= grok_step]
        
        if len(pre_grok) > 0 and len(post_grok) > 0:
            print(f"Pre-grok LLC: {pre_grok['llc_mean'].mean():.2f} ± {pre_grok['llc_std'].mean():.2f}")
            print(f"Post-grok LLC: {post_grok['llc_mean'].mean():.2f} ± {post_grok['llc_std'].mean():.2f}")
            print(f"LLC reduction: {(pre_grok['llc_mean'].mean() - post_grok['llc_mean'].mean()):.2f}")

if __name__ == '__main__':
    main()