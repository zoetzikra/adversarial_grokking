#!/usr/bin/env python3
"""
Quick test to verify MALA callback integration in calibration
"""

import torch
import numpy as np
from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k
from llc_calibrator import LLCCalibrator, LLCCalibratorConfig

def test_mala_integration():
    """Test if MALA data is preserved in calibration results"""
    print("Testing MALA callback integration in calibration...")
    
    # Setup minimal test
    config = config_resnet18_cifar10()
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Use a pretrained checkpoint for faster testing
    try:
        checkpoint_path = "/home/ztzifa/grok-adversarial/models/Fri_Jul_18_17:04:54_2025/checkpoint-s:500000.pt"
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded test checkpoint")
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}")
        print("   Using random weights for test")
    
    model.eval()
    
    # Load minimal data
    train_loader, _ = cifar10_dataloaders(config)
    
    # Setup calibrator with minimal settings for fast test
    calibrator_config = LLCCalibratorConfig(device=config.device)
    calibrator_config.calibration_epsilons = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    # calibrator_config.calibration_gammas = [1000.0]  # Only 1 gamma value
    
    llc_calibrator = LLCCalibrator(calibrator_config)
    
    print("Running mini calibration sweep...")
    
    # Option 1: Single gamma calibration (original method)
    print("\n=== SINGLE GAMMA CALIBRATION ===")
    try:
        optimal_params_single = llc_calibrator.calibrate_hyperparameters(
            model, 
            train_loader, 
            checkpoint_name="test_single",
            selection_method="mala",  # Try MALA selection
            use_tuned_beta=True
        )
        print("✓ Single-gamma calibration completed successfully")
        print(f"Selected parameters: ε={optimal_params_single['epsilon']:.2e}, γ={optimal_params_single['gamma']:.0f}")
        if optimal_params_single['mala_acceptance'] is not None:
            print(f"MALA acceptance: {optimal_params_single['mala_acceptance']:.3f}")
        
    except Exception as e:
        print(f"❌ Single-gamma calibration failed: {e}")
        optimal_params_single = None
    
    # Option 2: Multi-gamma calibration (new method)
    print("\n=== MULTI-GAMMA CALIBRATION ===")
    try:
        gamma_values = [1000.0, 5000.0, 10000.0] 
        optimal_params = llc_calibrator.calibrate_hyperparameters_multi_gamma(
            model=model,
            train_loader=train_loader,
            gamma_values=gamma_values,
            checkpoint_name="test_multi",
            use_tuned_beta=True
        )
        
        print("✓ Multi-gamma calibration completed successfully")
        print(f"🏆 BEST PARAMETERS (multi-gamma): ε={optimal_params['epsilon']:.2e}, γ={optimal_params['gamma']:.0f}")
        if optimal_params['mala_acceptance'] is not None:
            print(f"🏆 MALA acceptance: {optimal_params['mala_acceptance']:.3f}")
            
        # Compare with single-gamma result
        if optimal_params_single is not None:
            print(f"\n=== COMPARISON ===")
            print(f"Single-gamma: ε={optimal_params_single['epsilon']:.2e}, γ={optimal_params_single['gamma']:.0f}, MALA={optimal_params_single.get('mala_acceptance', 'N/A')}")
            print(f"Multi-gamma:  ε={optimal_params['epsilon']:.2e}, γ={optimal_params['gamma']:.0f}, MALA={optimal_params.get('mala_acceptance', 'N/A')}")
            
            if optimal_params.get('mala_acceptance') and optimal_params_single.get('mala_acceptance'):
                if optimal_params['mala_acceptance'] > optimal_params_single['mala_acceptance']:
                    print("🎉 Multi-gamma achieved better MALA acceptance!")
                else:
                    print("📊 Single-gamma had better MALA acceptance")
        
        # Check if MALA data was available
        if optimal_params.get('mala_acceptance') is not None:
            print("✅ MALA data found in multi-gamma results!")
        else:
            print("⚠️  No MALA data found in final results")
            
    except Exception as e:
        print(f"❌ Multi-gamma calibration failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to single-gamma result if available
        if 'optimal_params_single' in locals() and optimal_params_single is not None:
            print("📋 Using single-gamma result as fallback")
            optimal_params = optimal_params_single

if __name__ == "__main__":
    test_mala_integration()
