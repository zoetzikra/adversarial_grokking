# Adversarial Grokking Replication Guide

This guide provides comprehensive instructions for replicating the research on "Deep Networks Always Grok and Here is Why" - specifically the adversarial grokking experiments with ResNet18 on CIFAR-10.

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Running Experiments](#running-experiments)
6. [LLC Estimation](#llc-estimation)
7. [Analysis and Visualization](#analysis-and-visualization)
8. [Troubleshooting](#troubleshooting)
9. [File Structure](#file-structure)

## Project Overview

This project investigates grokking behavior in deep neural networks when trained on adversarial examples. The main components are:

- **Training**: ResNet18 models on CIFAR-10 with adversarial training
- **LLC Estimation**: Local Learning Coefficient estimation using SGLD sampling
- **Analysis**: Tracking grokking behavior through training dynamics
- **Visualization**: Tools for analyzing and plotting results

### Key Features
- Adversarial training with PGD attacks
- Local complexity approximation
- LLC estimation with calibration
- Weights & Biases integration for experiment tracking
- FFCV for fast data loading

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (H100, A100, or similar)
- **Memory**: 40GB+ RAM recommended
- **Storage**: 50GB+ free space for datasets and models

### Software Requirements
- **OS**: Linux (tested on Ubuntu/RHEL)
- **Python**: 3.8-3.10
- **CUDA**: 12.1+ (for GPU acceleration)
- **Slurm**: For cluster job submission (optional)

### Dependencies
- PyTorch 2.7.1+
- torchvision 0.22.1+
- FFCV (for fast data loading)
- devinterp (for LLC estimation)
- Weights & Biases (for experiment tracking)
- OpenCV (for FFCV compatibility)

## Installation

### 1. Environment Setup

```bash
# Load required modules (for HPC clusters)
module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1

# Create and activate conda environment
conda create -n devinterp_env python=3.10
conda activate devinterp_env
```

### 2. Install Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# Install other core dependencies
pip install ml_collections wandb tqdm matplotlib numpy pandas scipy
```

### 3. Install FFCV (Fast Data Loading)

FFCV requires OpenCV and can be tricky to install. Use the provided installation script:

```bash
# For HPC clusters with EasyBuild
sbatch installation_scripts/install_ffcv_fixed.job

# Or manually install OpenCV first, then FFCV
pip install opencv-python==4.8.1.78
pip install ffcv
```

### 4. Install devinterp (LLC Estimation)

```bash
pip install devinterp==1.3.2
```

### 5. Verify Installation

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import ffcv; print('FFCV: OK')
import devinterp; print('devinterp: OK')
import wandb; print('wandb: OK')
"
```

## Data Preparation

### CIFAR-10 Dataset

The project uses CIFAR-10 dataset. FFCV will automatically download and cache it:

```bash
# The dataset will be automatically downloaded on first run
# Default location: ~/.cache/ffcv/
```

### Data Preprocessing

The code handles data normalization automatically:
- **Normalized**: Mean=0, Std=1.25 (recommended for adversarial training)
- **Raw**: [0,1] range (alternative option)

## Running Experiments

### 1. Basic Training Run

```bash
# Activate environment
conda activate devinterp_env

# Set Weights & Biases API key
export WANDB_API_KEY=your_api_key_here

# Run training
python train_resnet18_cifar10.py
```

### 2. Using SLURM (HPC Clusters)

```bash
# Submit job using provided script
sbatch run_adversarial_grokking_ffcv.job
```

### 3. Custom Configuration

You can modify parameters via command line:

```bash
python train_resnet18_cifar10.py \
    --lr 1e-3 \
    --num_steps 500000 \
    --train_batch_size 256 \
    --compute_LC True \
    --compute_robust True \
    --atk_eps 50/255
```

### 4. Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `num_steps` | 500000 | Training steps |
| `train_batch_size` | 256 | Batch size |
| `compute_LC` | True | Compute local complexity |
| `compute_robust` | True | Compute adversarial robustness |
| `atk_eps` | 50/255 | PGD attack epsilon |
| `k` | 16 | ResNet width parameter |
| `use_ffcv` | True | Use FFCV for fast data loading |

## LLC Estimation

### 1. Understanding LLC

Local Learning Coefficient (LLC) measures the local complexity of the loss landscape around a point. It's estimated using SGLD (Stochastic Gradient Langevin Dynamics) sampling.

### 2. LLC Calibration

Before running LLC estimation, calibrate the hyperparameters:

```bash
python llc_calibration.py
```

This will:
- Test different epsilon values for SGLD
- Find optimal gamma (localization parameter)
- Save calibration results

### 3. LLC Estimation (Separate from Training)

LLC estimation is **not integrated into the training loop**. Instead, it's performed as a separate analysis step:

#### Retrospective Analysis
```bash
# Estimate LLC for saved model checkpoints
python llc_compute_retrospective.py
```


### 4. LLC Analysis

Use the provided analysis tools:

```bash
# Analyze LLC results
python llc_analysis.py

# Generate LLC plots
python llc_analyze_results_script.py
```

### 5. Workflow for LLC Analysis

1. **Train the model** using `train_resnet18_cifar10.py`
2. **Calibrate LLC parameters** using `llc_calibration.py`
3. **Estimate LLC** for saved checkpoints using `llc_compute_retrospective.py`
4. **Analyze results** using the analysis scripts or notebook

## Analysis and Visualization

### 1. Training Statistics

```bash
# Analyze training dynamics
python analyze_training_stats.py
```

### 2. Model Diagnosis

```bash
# Check model stability
python diagnose_model.py

# Diagnose model stability with detailed analysis
python diagnose_model_stability.py
```

### 3. Jupyter Notebook

For interactive analysis:

```bash
jupyter lab llc_measuring_grokking_example.ipynb
```

### 4. Weights & Biases Dashboard

Monitor experiments in real-time:
- Training loss and accuracy
- Adversarial robustness metrics
- LLC estimates
- Local complexity measures

## Troubleshooting

### Common Issues

#### 1. FFCV Installation Problems
```bash
# Try alternative installation
pip install --no-cache-dir ffcv

# Or use conda
conda install -c conda-forge ffcv
```

#### 2. CUDA Memory Issues
```bash
# Reduce batch size
python train_resnet18_cifar10.py --train_batch_size 128

# Or disable FFCV
python train_resnet18_cifar10.py --use_ffcv False
```

#### 3. LLC Estimation Failures
```bash
# Recalibrate LLC parameters
python llc_calibration.py

# Check device configuration
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Weights & Biases Issues
```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Or disable wandb logging
python train_resnet18_cifar10.py --wandb_log False
```

### Performance Optimization

#### 1. Data Loading
- Use FFCV for faster data loading
- Increase number of workers
- Use pinned memory

#### 2. LLC Estimation
- Reduce number of chains for faster estimation
- Use smaller epsilon values
- Adjust burn-in period

#### 3. Memory Management
- Use gradient checkpointing
- Reduce model width parameter `k`
- Use mixed precision training

## File Structure

```
grok-adversarial/
├── train_resnet18_cifar10.py      # Main training script
├── configs.py                     # Configuration management
├── models.py                      # ResNet18 model definition
├── dataloaders.py                 # Data loading utilities
├── attacks.py                     # PGD adversarial attacks
├── local_complexity.py            # Local complexity computation
├── llc_estimation.py              # LLC estimation utilities
├── samplers.py                    # Sampling utilities
├── utils.py                       # Utility functions
├── installation_scripts/          # Installation scripts
│   ├── install_ffcv_fixed.job    # FFCV installation
│   └── fix_ffcv_compatibility_numpy_downgrade.job
├── run_adversarial_grokking_ffcv.job  # SLURM job script
├── llc_calibration.py            # LLC calibration script
├── llc_analysis.py               # LLC analysis utilities
├── analyze_training_stats.py     # Training analysis
├── diagnose_model.py             # Model diagnosis
├── llc_measuring_grokking_example.ipynb  # Interactive notebook
├── data/                         # Dataset storage
├── models/                       # Model checkpoints
├── logs/                         # Training logs
├── analysis_output/              # Analysis results
└── wandb/                        # Weights & Biases logs
```

## Expected Results

### Training Dynamics
- Initial period of poor performance
- Sudden improvement (grokking event)
- Stabilization at high accuracy

### LLC Behavior
- High LLC during grokking transition
- Lower LLC in stable regions
- Correlation with loss landscape complexity

### Adversarial Robustness
- Initial vulnerability to attacks
- Gradual improvement in robustness
- Correlation with grokking events

## Citation

If you use this code in your research, please cite:

```bibtex
@article{humayun2024deep,
  title={Deep Networks Always Grok and Here is Why},
  author={Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
  journal={arXiv preprint arXiv:2402.15555},
  year={2024}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the Jupyter notebook examples
3. Check Weights & Biases logs for detailed error messages
4. Verify all dependencies are correctly installed

## Contributing

When contributing:
1. Test on clean environment
2. Update installation scripts if needed
3. Document new features
4. Maintain backward compatibility 