#!/usr/bin/env python3
"""
Diagnostic script to check model stability and find appropriate LLC parameters
"""

import torch as ch
import numpy as np
import argparse
from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k
from devinterp.utils import evaluate_ce, default_nbeta
from devinterp.optim.sgld import SGLD

def check_model_gradients(model, dataloader, device, config):
    """Check if model gradients are stable"""
    print("=== Checking Model Gradient Stability ===")
    
    model.eval()
    data_batch = next(iter(dataloader))
    inputs, targets = data_batch[0].to(device), data_batch[1].to(device)
    
    # Check initial loss
    with ch.no_grad():
        initial_loss = evaluate_ce(model, (inputs, targets))
        if isinstance(initial_loss, tuple):
            initial_loss = initial_loss[0]  # Extract loss from (loss, outputs) tuple
    print(f"Initial loss: {initial_loss:.4f}")
    
    # Check gradients for different epsilon values
    epsilon_test_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    
    for epsilon in epsilon_test_values:
        print(f"\nTesting epsilon = {epsilon:.2e}")
        
        # Create a copy of the model for testing
        test_model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn).to(device)
        test_model.load_state_dict(model.state_dict())
        
        # Add small perturbation
        with ch.no_grad():
            for param in test_model.parameters():
                param.add_(ch.randn_like(param) * epsilon)
        
        # Check if loss is finite
        try:
            with ch.no_grad():
                perturbed_loss = evaluate_ce(test_model, (inputs, targets))
                if isinstance(perturbed_loss, tuple):
                    perturbed_loss = perturbed_loss[0]
            
            if ch.isfinite(perturbed_loss):
                print(f"  ✓ Loss after perturbation: {perturbed_loss:.4f}")
                print(f"  ✓ Loss difference: {abs(perturbed_loss - initial_loss):.4f}")
            else:
                print(f"  ✗ Loss is not finite: {perturbed_loss}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_sgld_stability(model, dataloader, device):
    """Test SGLD stability with very conservative parameters"""
    print("\n=== Testing SGLD Stability ===")
    
    nbeta = default_nbeta(dataloader)
    print(f"Default nbeta: {nbeta:.3f}")
    
    # Test with extremely small epsilon values
    epsilon_values = [1e-9, 1e-8, 1e-7, 1e-6]
    gamma_values = [0.1, 0.5, 1.0]
    
    for epsilon in epsilon_values:
        for gamma in gamma_values:
            print(f"\nTesting ε={epsilon:.2e}, γ={gamma:.1f}")
            
            try:
                # Create optimizer
                optimizer = SGLD(
                    model.parameters(),
                    lr=epsilon,
                    localization=gamma,
                    nbeta=nbeta
                )
                
                # Get a batch
                data_batch = next(iter(dataloader))
                inputs, targets = data_batch[0].to(device), data_batch[1].to(device)
                
                # Try a few steps
                for step in range(5):
                    optimizer.zero_grad()
                    loss = evaluate_ce(model, (inputs, targets))
                    if isinstance(loss, tuple):
                        loss = loss[0]
                    loss.backward()
                    
                    # Check gradients
                    grad_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.norm().item() ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    if ch.isfinite(loss) and ch.isfinite(ch.tensor(grad_norm)):
                        print(f"  Step {step}: Loss={loss:.4f}, GradNorm={grad_norm:.4f}")
                    else:
                        print(f"  Step {step}: ✗ NaN detected")
                        break
                else:
                    print(f"  ✓ SGLD stable for 5 steps")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")

def suggest_parameters(model, dataloader, device):
    """Suggest appropriate parameters based on model characteristics"""
    print("\n=== Parameter Suggestions ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Check parameter magnitudes
    param_norms = []
    for param in model.parameters():
        param_norms.append(param.norm().item())
    
    avg_param_norm = np.mean(param_norms)
    max_param_norm = np.max(param_norms)
    min_param_norm = np.min(param_norms)
    
    print(f"Parameter norms - Avg: {avg_param_norm:.4f}, Min: {min_param_norm:.4f}, Max: {max_param_norm:.4f}")
    
    # Suggest epsilon based on parameter scale
    suggested_epsilon = min_param_norm * 1e-6
    print(f"\nSuggested epsilon range:")
    print(f"  Conservative: {suggested_epsilon:.2e} to {suggested_epsilon*10:.2e}")
    print(f"  Moderate: {suggested_epsilon*10:.2e} to {suggested_epsilon*100:.2e}")
    
    # Suggest gamma
    print(f"\nSuggested gamma values:")
    print(f"  Very conservative: 0.1, 0.5")
    print(f"  Conservative: 1.0, 2.0")
    print(f"  Moderate: 5.0, 10.0")

def main():
    parser = argparse.ArgumentParser(description='Diagnose model stability for LLC estimation')
    parser.add_argument('--checkpoint_path', required=True, 
                       help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    
    # Load config and model
    config = config_resnet18_cifar10()
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Load trained weights
    checkpoint = ch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    train_loader, _ = cifar10_dataloaders(config)
    
    device = 'cuda' if ch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Model loaded from: {args.checkpoint_path}")
    print(f"Device: {device}")
    
    # Run diagnostics
    check_model_gradients(model, train_loader, device, config)
    test_sgld_stability(model, train_loader, device)
    suggest_parameters(model, train_loader, device)
    
    print("\n=== Recommendations ===")
    print("1. Try epsilon values in the 1e-9 to 1e-7 range")
    print("2. Use very small gamma values (0.1 to 1.0)")
    print("3. Consider reducing the number of draws to 50-100")
    print("4. Increase burn-in steps to 90% of total")
    print("5. If still failing, the model may be too well-trained for stable LLC estimation")

if __name__ == '__main__':
    main() 