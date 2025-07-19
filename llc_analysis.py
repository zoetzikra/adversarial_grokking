import matplotlib.pyplot as plt
import numpy as np

def plot_llc_vs_local_complexity(stats, save_path='llc_comparison.png'):
    """Compare LLC with existing local complexity measure"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    steps = stats['train_step']
    
    # Plot 1: LLC evolution
    ax1.plot(steps, stats['llc_mean'], 'g-', label='LLC Mean', linewidth=2)
    ax1.fill_between(steps, 
                     np.array(stats['llc_mean']) - np.array(stats['llc_std']),
                     np.array(stats['llc_mean']) + np.array(stats['llc_std']),
                     alpha=0.3, color='green')
    ax1.set_ylabel('LLC Estimate')
    ax1.set_title('Local Learning Coefficient Evolution')
    ax1.legend()
    
    # Plot 2: Existing local complexity  
    train_lc = [lc.sum(1).mean(0).item() for lc in stats['train_LC']]
    test_lc = [lc.sum(1).mean(0).item() for lc in stats['test_LC']]
    
    ax2.plot(steps, train_lc, 'b-', label='Train LC')
    ax2.plot(steps, test_lc, 'r-', label='Test LC') 
    ax2.set_ylabel('Local Complexity (Linear Regions)')
    ax2.set_title('Existing Local Complexity Measure')
    ax2.legend()
    
    # Plot 3: Accuracy and robustness
    ax3.plot(steps, stats['test_acc'], 'k-', label='Test Accuracy')
    ax3.plot(steps, stats['adv_acc'], 'orange', label='Adversarial Accuracy')
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Training Steps')
    ax3.set_title('Performance Metrics')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    return fig

def analyze_grokking_phases(stats):
    """Identify grokking phases using both complexity measures"""
    steps = np.array(stats['train_step'])
    llc = np.array(stats['llc_mean'])
    local_complexity = np.array([lc.sum(1).mean(0).item() for lc in stats['train_LC']])
    test_acc = np.array(stats['test_acc'])
    adv_acc = np.array(stats['adv_acc'])
    
    # Find phase transitions
    # Phase 1: Memorization (test acc low, complexity high)
    # Phase 2: Grokking (test acc rises, complexity drops)
    # Phase 3: Robust (adv acc rises, complexity stabilizes)
    
    memorization_end = np.where(test_acc > 0.6)[0]
    if len(memorization_end) > 0:
        memorization_end = memorization_end[0]
    else:
        memorization_end = len(steps) // 2
        
    grokking_end = np.where(adv_acc > 0.5)[0]
    if len(grokking_end) > 0:
        grokking_end = grokking_end[0]
    else:
        grokking_end = len(steps)
    
    print(f"Phase Analysis:")
    print(f"Memorization: steps 0 to {steps[memorization_end]}")
    print(f"Grokking: steps {steps[memorization_end]} to {steps[min(grokking_end, len(steps)-1)]}")
    print(f"Robust: steps {steps[min(grokking_end, len(steps)-1)]} onwards")
    
    return {
        'memorization_end': memorization_end,
        'grokking_end': grokking_end,
        'phases': {
            'memorization': (0, memorization_end),
            'grokking': (memorization_end, grokking_end),
            'robust': (grokking_end, len(steps))
        }
    }