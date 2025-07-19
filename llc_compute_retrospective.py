import torch
import glob
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k
from llc_estimation import LLCEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, 
                       help='Directory containing saved checkpoints')
    parser.add_argument('--llc_epsilon', type=float, default=0.03,
                       help='Calibrated epsilon value')
    parser.add_argument('--llc_gamma', type=float, default=5.0,
                       help='Calibrated gamma value')
    args = parser.parse_args()
    
    # Load config and setup
    config = config_resnet18_cifar10()
    config.llc_epsilon = args.llc_epsilon
    config.llc_gamma = args.llc_gamma
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint-s:*.pt'))
    checkpoint_files = [f for f in checkpoint_files if 'checkpoint-s:-1.pt' not in f]  # Skip base model
    
    # Sort by training step
    def extract_step(filename):
        return int(filename.split('checkpoint-s:')[1].split('.pt')[0])
    
    checkpoint_files = sorted(checkpoint_files, key=extract_step)
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Setup model and data
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    train_loader, _ = cifar10_dataloaders(config)
    llc_dataloader = DataLoader(
        list(train_loader.dataset)[:1000],  # Subset for efficiency
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    
    # Initialize LLC estimator
    llc_estimator = LLCEstimator(config)
    
    # Compute LLC for each checkpoint
    results = []
    for checkpoint_file in tqdm(checkpoint_files, desc="Computing LLC"):
        step = extract_step(checkpoint_file)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Compute LLC
        llc_mean, llc_std = llc_estimator.estimate_llc(model, llc_dataloader)
        
        results.append({
            'step': step,
            'llc_mean': llc_mean,
            'llc_std': llc_std,
            'checkpoint_file': checkpoint_file
        })
        
        print(f"Step {step}: LLC = {llc_mean:.2f} ± {llc_std:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(args.checkpoint_dir, 'llc_retrospective_results.csv')
    results_df.to_csv(output_file, index=False)
    
    # Also save as pickle for easy loading
    torch.save(results, os.path.join(args.checkpoint_dir, 'llc_retrospective_results.pt'))
    
    print(f"Results saved to {output_file}")
    print(f"Use analyze_llc_results.py to visualize the results")

if __name__ == '__main__':
    main()