import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k
import numpy as np
import random
from llc_calibrator import LLCCalibrator, LLCCalibratorConfig

def set_reproducible_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, 
                       help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    set_reproducible_seeds(42)
    
    # Load config and model
    config = config_resnet18_cifar10()
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Load trained weights
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract checkpoint name for plotting
    checkpoint_name = Path(args.checkpoint_path).stem
    
    # Load data
    train_loader, _ = cifar10_dataloaders(config)
    
    
    # llc_estimator = LLCEstimator(config)
    calibrator_config = LLCCalibratorConfig(device=config.device)
    llc_calibrator = LLCCalibrator(calibrator_config)
    print("Calibrating LLC hyperparameters on trained model...")
    # analyzer = llc_estimator.calibrate_hyperparameters(model, calibration_loader, checkpoint_name=checkpoint_name)
    optimal_params = llc_calibrator.calibrate_hyperparameters(model, train_loader, checkpoint_name=checkpoint_name)
    

    # Estimate LLC with optimal parameters
    print("Estimating LLC with optimal parameters...")
    llc_results = llc_calibrator.estimate_llc(
        model, 
        train_loader, 
        hyperparams=optimal_params,
        save_path=f"./llc_calibration_results/{checkpoint_name}/final_llc_estimation",
        seed=43  # Use different seed for final estimation to avoid exact duplication
    )
    
    # Plot the final LLC estimation trace (this is what you want!)
    if 'loss/trace' in llc_results:
        final_trace_path = f"./llc_calibration_results/{checkpoint_name}/final_llc_trace.png"
        llc_calibrator.plot_sampling_evolution(
            llc_results, 
            save_path=final_trace_path, 
            show=False
        )
        print(f"Final LLC trace plot saved to {final_trace_path}")
        print(f"Final trace shape: {llc_results['loss/trace'].shape}")
    
    print("Calibration complete. Review plots and update config accordingly.")

if __name__ == '__main__':
    main()