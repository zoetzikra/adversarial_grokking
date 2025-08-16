#!/usr/bin/env python3
"""
Diagnose model behavior to understand why LLC estimation is failing
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k

def check_model_gradients(model, dataloader, device):
    """Check if model gradients are finite and reasonable"""
    
    print("=== CHECKING MODEL GRADIENTS ===")
    
    model.eval()
    model.zero_grad()
    
    # Get a single batch
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= 1:  # Only check first batch
            break
            
        data = data.to(device)
        target = target.to(device)
        
        print(f"Input shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Input range: [{data.min():.4f}, {data.max():.4f}]")
        
        # Forward pass
        output = model(data)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Check for NaN/Inf in output
        if torch.isnan(output).any():
            print("⚠️  NaN detected in model output!")
            return False
        if torch.isinf(output).any():
            print("⚠️  Inf detected in model output!")
            return False
            
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        print(f"Loss: {loss.item():.4f}")
        
        if torch.isnan(loss):
            print("⚠️  NaN loss!")
            return False
        if torch.isinf(loss):
            print("⚠️  Inf loss!")
            return False
            
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_params = 0
        finite_grads = 0
        nan_grads = 0
        inf_grads = 0
        zero_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                total_params += param.numel()
                
                grad = param.grad
                finite_grads += torch.isfinite(grad).sum().item()
                nan_grads += torch.isnan(grad).sum().item()
                inf_grads += torch.isinf(grad).sum().item()
                zero_grads += (grad == 0).sum().item()
                
                # Check gradient norms for first few layers
                if 'conv1' in name or 'layer1' in name:
                    grad_norm = grad.norm().item()
                    print(f"  {name}: grad_norm = {grad_norm:.6f}")
        
        print(f"\nGradient Statistics:")
        print(f"  Total parameters: {total_params}")
        print(f"  Finite gradients: {finite_grads}")
        print(f"  NaN gradients: {nan_grads}")
        print(f"  Inf gradients: {inf_grads}")
        print(f"  Zero gradients: {zero_grads}")
        
        if nan_grads > 0:
            print("⚠️  NaN gradients detected!")
            return False
        if inf_grads > 0:
            print("⚠️  Inf gradients detected!")
            return False
            
        print("✅ All gradients are finite")
        return True

def check_model_parameters(model):
    """Check if model parameters are reasonable"""
    
    print("\n=== CHECKING MODEL PARAMETERS ===")
    
    total_params = 0
    finite_params = 0
    nan_params = 0
    inf_params = 0
    zero_params = 0
    
    param_norms = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        finite_params += torch.isfinite(param).sum().item()
        nan_params += torch.isnan(param).sum().item()
        inf_params += torch.isinf(param).sum().item()
        zero_params += (param == 0).sum().item()
        
        param_norm = param.norm().item()
        param_norms.append(param_norm)
        
        # Show first few layers
        if len(param_norms) <= 5:
            print(f"  {name}: norm = {param_norm:.6f}, shape = {param.shape}")
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params}")
    print(f"  Finite parameters: {finite_params}")
    print(f"  NaN parameters: {nan_params}")
    print(f"  Inf parameters: {inf_params}")
    print(f"  Zero parameters: {zero_params}")
    print(f"  Parameter norm range: [{min(param_norms):.6f}, {max(param_norms):.6f}]")
    
    if nan_params > 0:
        print("⚠️  NaN parameters detected!")
        return False
    if inf_params > 0:
        print("⚠️  Inf parameters detected!")
        return False
        
    print("✅ All parameters are finite")
    return True

def test_simple_forward_backward(model, dataloader, device):
    """Test simple forward/backward with different learning rates"""
    
    print("\n=== TESTING SIMPLE FORWARD/BACKWARD ===")
    
    model.train()
    
    # Get a single batch
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= 1:
            break
            
        data = data.to(device)
        target = target.to(device)
        
        # Test different learning rates
        lr_values = [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1]
        
        for lr in lr_values:
            print(f"\nTesting lr = {lr:.1e}")
            
            # Reset gradients
            model.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            if torch.isnan(loss):
                print(f"  ✗ NaN loss")
                continue
            if torch.isinf(loss):
                print(f"  ✗ Inf loss")
                continue
                
            # Backward pass
            loss.backward()
            
            # Check gradients
            has_nan_grad = False
            has_inf_grad = False
            
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
            
            if has_nan_grad:
                print(f"  ✗ NaN gradients")
            elif has_inf_grad:
                print(f"  ✗ Inf gradients")
            else:
                print(f"  ✓ Finite gradients")
                
                # Try a small parameter update
                try:
                    with torch.no_grad():
                        for param in model.parameters():
                            if param.grad is not None:
                                param -= lr * param.grad
                    
                    # Check if parameters are still finite
                    has_nan_param = False
                    has_inf_param = False
                    
                    for param in model.parameters():
                        if torch.isnan(param).any():
                            has_nan_param = True
                        if torch.isinf(param).any():
                            has_inf_param = True
                    
                    if has_nan_param:
                        print(f"  ✗ NaN parameters after update")
                    elif has_inf_param:
                        print(f"  ✗ Inf parameters after update")
                    else:
                        print(f"  ✓ Parameters remain finite after update")
                        
                except Exception as e:
                    print(f"  ✗ Error during update: {e}")

def main():
    # Load config and model
    config = config_resnet18_cifar10()
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Load trained weights
    checkpoint_path = "models/Fri_Jul_18_17:04:54_2025/checkpoint-s:5277.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    train_loader, _ = cifar10_dataloaders(config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Model type: {type(model)}")
    
    # Run diagnostics
    param_ok = check_model_parameters(model)
    grad_ok = check_model_gradients(model, train_loader, device)
    test_simple_forward_backward(model, train_loader, device)
    
    print(f"\n=== SUMMARY ===")
    print(f"Parameters OK: {param_ok}")
    print(f"Gradients OK: {grad_ok}")
    
    if param_ok and grad_ok:
        print("✅ Model appears healthy")
        print("The issue might be with SGLD implementation or devinterp")
    else:
        print("⚠️  Model has issues that need to be fixed")

if __name__ == "__main__":
    main() 