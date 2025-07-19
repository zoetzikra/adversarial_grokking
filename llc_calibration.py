import torch
from torch.utils.data import DataLoader
import argparse
from configs import config_resnet18_cifar10
from dataloaders import cifar10_dataloaders
from models import make_resnet18k

from llc_estimation import LLCEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, 
                       help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    # Load config and model
    config = config_resnet18_cifar10()
    model = make_resnet18k(k=config.k, num_classes=config.num_class, bn=config.use_bn)
    
    # Load trained weights
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    train_loader, _ = cifar10_dataloaders(config)
    calibration_loader = DataLoader(
        list(train_loader.dataset)[:1000],  # Subset for speed
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    
    # Initialize estimator and calibrate
    llc_estimator = LLCEstimator(config)
    print("Calibrating LLC hyperparameters on trained model...")
    analyzer = llc_estimator.calibrate_hyperparameters(model, calibration_loader)
    
    print("Calibration complete. Review plots and update config accordingly.")

if __name__ == '__main__':
    main()