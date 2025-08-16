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
    calibrator_config.calibration_epsilons = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    calibrator_config.calibration_gammas = [1000.0]  # Only 1 gamma value
    
    llc_calibrator = LLCCalibrator(calibrator_config)
    
    print("Running mini calibration sweep...")
    try:
        optimal_params = llc_calibrator.calibrate_hyperparameters(
            model, 
            train_loader, 
            checkpoint_name="test",
            selection_method="mala",  # Try MALA selection
            use_tuned_beta=True
        )
        
        print("✓ Calibration completed successfully")
        print(f"Selected parameters: {optimal_params}")
        
        # Check if MALA data was available
        if 'mala_acceptance' in optimal_params or any('mala' in str(k) for k in optimal_params.keys()):
            print("✓ MALA data found in results!")
        else:
            print("⚠️  No MALA data found in final results")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mala_integration()
