import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce, default_nbeta
from devinterp.vis_utils import EpsilonBetaAnalyzer
import os
import warnings
import signal
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

class LLCEstimator:
    def __init__(self, config):
        self.device = config.device
        self.epsilon = config.llc_epsilon  # SGLD step size
        self.gamma = config.llc_gamma      # localization parameter 
        self.beta = config.llc_beta        # inverse temperature parameter
        self.num_chains = config.llc_num_chains
        self.num_draws = config.llc_num_draws
        self.num_burnin = config.llc_num_burnin
        self.seed = 42
        
    def _set_seeds(self):
        """Set consistent random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Set random seeds to {self.seed}")

    def _validate_model_state(self, model):
        """Validate model state and ensure gradients are enabled"""
        print("Validating model state...")
        
        # Check for NaN/Inf values in model parameters
        has_nan = False
        has_inf = False
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                if torch.isnan(param).any():
                    has_nan = True
                    print(f"⚠️  NaN found in parameter: {name}")
                if torch.isinf(param).any():
                    has_inf = True
                    print(f"⚠️  Inf found in parameter: {name}")
        
        if has_nan or has_inf:
            print("❌ Model contains NaN/Inf values - cannot proceed with LLC estimation")
            return False
            
        # Note: Gradient enabling is now handled in robust_evaluate_model
        # to ensure it happens at the right time for LLC estimation
            
        # Verify model state
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model validation passed")
        print(f"✓ Total parameters: {total_params:,}")
        
        # Store model info for auto-scaling
        self._num_parameters = total_params
        self._model_loaded = True
        
        return True
    
    def robust_evaluate_model(self, model, data):
        """Robust evaluation function that handles model state properly"""
        # Handle both dataloader and data tuple inputs
        if hasattr(data, '__iter__') and not isinstance(data, (tuple, list)):
            # It's a dataloader - extract a batch
            try:
                data_batch = next(iter(data))
            except StopIteration:
                print("❌ Empty dataloader")
                return torch.tensor(float('inf')), {}
            inputs, targets = data_batch
        else:
            # It's already a data tuple (inputs, targets)
            inputs, targets = data
        
        # Ensure model is in evaluation mode
        model.eval()
        
        # CRITICAL: Ensure all parameters require gradients for LLC estimation
        for param in model.parameters():
            param.requires_grad_(True)
        
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Fix device mismatch: ensure model is on the same device as data
        model_device = next(model.parameters()).device
        target_device = torch.device(self.device)
        
        if model_device != target_device:
            # Only print warning once per model, not every evaluation
            if not hasattr(model, '_device_warning_printed'):
                print(f"  Moving model from {model_device} to {target_device}")
                model._device_warning_printed = True
            model = model.to(target_device)
        
        try:
            # Forward pass - NO torch.no_grad() - we need gradients for LLC estimation!
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # Return the loss tensor (not loss.item()) so devinterp can handle it properly
            return loss, {"logits": outputs}
            
        except Exception as e:
            print(f"⚠️  Error in model evaluation: {e}")
            # Return a high loss tensor to indicate failure
            return torch.tensor(float('inf'), device=self.device), {}
        
    def estimate_llc_outdated(self, model, dataloader, verbose=False):
        """Estimate the Local Learning Coefficient (LLC) for a model."""
        self._validate_model_state(model)
        
        # Use self.beta if set, otherwise fall back to default_nbeta
        beta_param = self.beta if self.beta is not None else default_nbeta
        
        if verbose:
            print(f"Using beta = {beta_param} for LLC estimation")
        
        # Use robust evaluation function
        stats = estimate_learning_coeff_with_summary(
            model=model,
            evaluate=self.robust_evaluate_model,
            loader=dataloader,
            epsilon=self.epsilon,
            beta=beta_param,
            localization=self.gamma,
            num_chains=self.num_chains,
            num_draws=self.num_draws,
            num_burnin_steps=self.num_burnin,
            verbose=verbose
        )
        
        # Extract LLC values, handling both aggregated and individual cases
        llc_mean, llc_std = self._extract_llc_values(stats)
        
        return llc_mean, llc_std

    def _extract_llc_values(self, stats):
        """Extract LLC mean and std from stats dictionary."""
        # Try aggregated values first
        llc_mean = stats.get('llc/mean')
        llc_std = stats.get('llc/std')
        
        if llc_mean is not None and llc_std is not None:
            return llc_mean, llc_std
        
        # Calculate from individual values if aggregated not found
        llc_means = stats.get('llc/means')
        if llc_means is not None:
            return np.mean(llc_means), np.std(llc_means)
        
        return np.nan, np.nan
    
    def calibrate_hyperparameters(self, model, dataloader, checkpoint_name=None):
        """Calibrate hyperparameters using EpsilonBetaAnalyzer with robust parameter selection"""
        print("Starting hyperparameter calibration...")
        print("Using stability-based parameter selection for robust results")
        
        self._set_seeds()
        self._validate_model_state(model)
        
        '''to be removed            FROM HERE'''
        # Add this block only
        data_info = {
            'dataloader_length': len(dataloader),
            'batch_size': dataloader.batch_size,
            'seed': self.seed
        }
        data_info_path = f"calibration_data_info_{checkpoint_name}.json"
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        print(f"Saved calibration data info to {data_info_path}")
        '''to be removed            TO HERE'''

        # Adaptive epsilon ranges based on model complexity and convergence
        # Start with a wide range and let the analyzer find the best
        epsilon_range = [1e-15, 5e-15, 1e-14, 5e-14, 1e-13, 5e-13, 1e-12, 5e-12, 1e-11, 5e-11, 1e-10, 5e-10, 1e-9, 5e-9, 1e-8]
        gamma_range = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        print(f"Using adaptive epsilon range: {[f'{e:.1e}' for e in epsilon_range]}")
        print(f"Using gamma range: {gamma_range}")
        
        analyzer = EpsilonBetaAnalyzer()
        
        # Configure the sweep with comprehensive ranges
        analyzer.configure_sweep(
            llc_estimator=self._estimate_llc_for_calibration,
            llc_estimator_kwargs=dict(
                model=model,
                # evaluate=evaluate_ce,
                evaluate=self.robust_evaluate_model,
                device=self.device,
                loader=dataloader,
            ),
            min_epsilon=epsilon_range[0],
            max_epsilon=epsilon_range[-1],
            epsilon_samples=len(epsilon_range),
            min_beta=0.4,      # Minimum beta (avoid too high temperature)
            max_beta=45000.0,  # Maximum beta (avoid too low temperature) 
            beta_samples=10,     # Increased from 3 to 5 for better coverage
            dataloader=dataloader,
        )
        
        # Run the sweep
        analyzer.sweep()
        
        # Extract best parameters from analyzer
        best_params = None
        # The analyzer stores results in sweep_df after sweep()
        if analyzer.sweep_df is not None and len(analyzer.sweep_df) > 0:
            print(f"Found {len(analyzer.sweep_df)} successful calibration runs")
            
            # Calculate additional statistics if not already present
            if 'llc/final' not in analyzer.sweep_df.columns:
                analyzer.sweep_df['llc/final'] = analyzer.sweep_df['llc/trace'].apply(
                    lambda x: x[:, -50:].mean() if len(x.shape) == 2 else x[-50:].mean()
                )
            
            if 'llc/std_over_mean' not in analyzer.sweep_df.columns:
                analyzer.sweep_df['llc/std_over_mean'] = analyzer.sweep_df['llc/trace'].apply(
                    lambda x: x[:, -50:].std() / x[:, -50:].mean() if len(x.shape) == 2 else x[-50:].std() / x[-50:].mean()
            )
            
            # Filter out NaN results
            valid_results = analyzer.sweep_df.dropna(subset=['llc/final'])
            
            if len(valid_results) > 0:
                print(f"\n=== HYPERPARAMETER SELECTION ===")
                
                # Debug: Check for negative LLC values
                print(f"LLC value range: {valid_results['llc/final'].min():.4f} to {valid_results['llc/final'].max():.4f}")
                print(f"Stability range: {valid_results['llc/std_over_mean'].min():.4f} to {valid_results['llc/std_over_mean'].max():.4f}")
                
                # Check for negative LLC values (failure mode according to guide)
                negative_llc_count = (valid_results['llc/final'] < 0).sum()
                if negative_llc_count > 0:
                    print(f"⚠️  WARNING: {negative_llc_count}/{len(valid_results)} runs have negative LLC values")
                    print("   This indicates failure modes: step size too large, chain too long, or w* not near local minimum")
                    print("   Consider: reducing step size, shortening chain, increasing γ, or checking model convergence")
                
                # Filter out results with NaN or extreme values
                clean_results = valid_results.dropna(subset=['llc/final', 'llc/std_over_mean'])
                clean_results = clean_results[clean_results['llc/final'].abs() < 1000]  # Remove extreme values
                
                if len(clean_results) == 0:
                    print("Warning: No clean results found, using all results...")
                    clean_results = valid_results
                
                # Separate positive and negative LLC results
                positive_llc = clean_results[clean_results['llc/final'] > 0]
                negative_llc = clean_results[clean_results['llc/final'] < 0]
                
                print(f"Results with positive LLC: {len(positive_llc)}")
                print(f"Results with negative LLC: {len(negative_llc)}")
                
                # Robust parameter selection based on stability
                print("Using stability-based parameter selection")
                
                # Prefer positive LLC with good stability
                if len(positive_llc) > 0:
                    # Find the most stable positive LLC result
                    best_stability_idx = positive_llc['llc/std_over_mean'].abs().idxmin()
                    best_row = positive_llc.loc[best_stability_idx]
                    print("✓ Using positive LLC result with best stability")
                else:
                    # Fall back to best stability among all results
                    best_stability_idx = clean_results['llc/std_over_mean'].abs().idxmin()
                    best_row = clean_results.loc[best_stability_idx]
                    print("⚠️  No positive LLC found, using best stability among all results")
                
                best_params = {
                    'epsilon': best_row['epsilon'],
                    'beta': best_row['beta'],
                    'llc_mean': best_row['llc/final'],
                    'llc_std_over_mean': best_row['llc/std_over_mean']
                }
                
                print(f"\nSelected parameters (best stability):")
                print(f"  ε = {best_params['epsilon']:.2e}")
                print(f"  β = {best_params['beta']:.3f}")
                print(f"  LLC = {best_params['llc_mean']:.4f}")
                print(f"  Stability (std/mean) = {best_params['llc_std_over_mean']:.4f}")
                
                if best_params['llc_mean'] < 0:
                    print(f"  ⚠️  WARNING: Negative LLC selected - may need parameter adjustment")
                
                # Also print top 5 results for comparison
                print(f"\nTop 5 parameter combinations (by stability):")
                # Create a temporary column for sorting by absolute stability
                clean_results['abs_stability'] = clean_results['llc/std_over_mean'].abs()
                top_5 = clean_results.nsmallest(5, 'abs_stability')
                for i, (_, row) in enumerate(top_5.iterrows()):
                    llc_sign = "⚠️" if row['llc/final'] < 0 else "✓"
                    print(f"  {i+1}. {llc_sign} ε={row['epsilon']:.2e}, β={row['beta']:.3f}, LLC={row['llc/final']:.4f}, stability={row['llc/std_over_mean']:.4f}")
                # Clean up temporary column
                clean_results = clean_results.drop('abs_stability', axis=1)
                
                # Save detailed results for analysis
                csv_filename = f'llc_calibration_detailed_results_{checkpoint_name}.csv'
                clean_results.to_csv(csv_filename, index=False)
                print(f"\nDetailed results saved to: {csv_filename}")
            else:
                print("No valid results found in analyzer")
        else:
            print("No sweep results available")
        
        # Save the plots
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        analyzer.plot()
        plot_filename = f'llc_calibration_results_{checkpoint_name}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Calibration plots saved to {plot_filename}")
        
        # If we got best parameters, do a single test run
        if best_params is not None:
                # Add these lines only
                model_state_path = f"calibration_model_state_{checkpoint_name}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_params': best_params,
                    'seed': self.seed
                }, model_state_path)
                print(f"Saved calibration model state to {model_state_path}")

                print(f"\nBest parameters from analyzer: {best_params}")
                
                hyperparams_filename = f'best_hyperparameters_{checkpoint_name}.json'
                
                # Add metadata to the saved parameters
                hyperparams_to_save = {
                    'checkpoint_name': checkpoint_name,
                    'model_class': model.__class__.__name__,
                    'best_hyperparameters': best_params,
                    'calibration_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'config_info': {
                        'llc_epsilon': self.epsilon,
                        'llc_gamma': self.gamma,
                        'llc_beta': self.beta,
                        'llc_num_chains': self.num_chains,
                        'llc_num_draws': self.num_draws,
                        'llc_num_burnin': self.num_burnin
                    }
                }
                
                with open(hyperparams_filename, 'w') as f:
                    json.dump(hyperparams_to_save, f, indent=2, default=str)
                
                print(f"✓ Best hyperparameters saved to: {hyperparams_filename}")
                
                # Also print recommended config values
                print(f"\n=== RECOMMENDED CONFIG VALUES ===")
                print(f"Update your config with:")
                print(f"  llc_epsilon = {best_params['epsilon']:.2e}")
                print(f"  # Note: nbeta = {best_params['beta']:.3f} (this goes in optimizer_kwargs)")
                
                self._test_best_parameters(model, dataloader, best_params, checkpoint_name)
                
                # Print summary of generated files
                print(f"\n=== GENERATED FILES SUMMARY ===")
                if checkpoint_name:
                    print(f"✓ Calibration plots: llc_calibration_results_{checkpoint_name}.png")
                    print(f"✓ Detailed results: llc_calibration_detailed_results_{checkpoint_name}.csv")
                    print(f"✓ Best hyperparameters: best_hyperparameters_{checkpoint_name}.json")
                else:
                    print(f"✓ Calibration plots: llc_calibration_results.png")
                    print(f"✓ Detailed results: llc_calibration_detailed_results.csv")
                    print(f"✓ Best hyperparameters: best_hyperparameters.json")
                print(f"\nAll files saved in current directory.")
        else:
            print("No best parameters extracted from analyzer")
            print("Model may be too converged for stable LLC estimation")
            print("Consider:")
            print("  1. Using an earlier checkpoint (less sharp minimum)")
            print("  2. Using even smaller epsilon values (e.g., 1e-16)")
            print("  3. Using larger gamma values (e.g., 50-100)")
            print("  4. Checking model stability with diagnostic script")
            
        
        return analyzer
    
    def _test_best_parameters(self, model, dataloader, best_params, checkpoint_name=None):
        """Test the best parameters with a single run and plot trace"""
        print("\n=== TESTING BEST PARAMETERS ===")
        self._set_seeds()

        model_state_path = f"calibration_model_state_{checkpoint_name}.pt"
        if os.path.exists(model_state_path):
            checkpoint = torch.load(model_state_path, map_location=self.device, weights_only=False)

            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded calibration model state from {model_state_path}")
            
            # Verify hyperparameters match
            if checkpoint['best_params'] != best_params:
                print("ERROR: Hyperparameters don't match calibration!")
                print(f"Calibration: {checkpoint['best_params']}")
                print(f"Testing: {best_params}")
            else:
                print("Hyperparameters verified ✓")
        else:
            print(f"WARNING: No calibration model state found at {model_state_path}")
        

        # Validate model state
        self._validate_model_state(model)
        # Debug: Print the raw best_params
        print(f"Raw best_params: {best_params}")
        
        # Extract best parameters
        best_epsilon = best_params['epsilon']
        best_beta = best_params['beta']
        
        # Validate step size to prevent SGLD wandering (per devinterp docs)
        if best_epsilon > 1e-6:
            print(f"⚠️  WARNING: Step size {best_epsilon:.2e} may be too large, consider reducing")
        if best_epsilon < 1e-15:
            print(f"⚠️  WARNING: Step size {best_epsilon:.2e} may be too small for effective sampling")
        
        # Use a larger gamma to provide stronger localization and prevent negative LLC
        # Based on devinterp documentation: small gamma can cause SGLD to wander too far
        gamma = 5.0  # Increased from 1.0 to provide better localization
        
        # Use the calibrated beta value as nbeta
        nbeta = best_beta
        # nbeta = default_nbeta(dataloader)
        print(f"DEFAULT Nbeta: {nbeta}")
        
        print(f"Debug: Using gamma = {gamma} (hardcoded to match calibration)")
        print(f"Debug: Using num_chains = {self.num_chains}, num_draws = {self.num_draws}, num_burnin = {self.num_burnin}")
        
        # Test evaluation function first
        try:
            # Validate model state before testing
            if not self._validate_model_state(model):
                print("❌ Model validation failed - cannot test parameters")
                return None
                
            # Make a single test run with chosen hyperparameters
            print("Making single test run with best hyperparameters...")
            
            # Debug: Check dataloader state
            print(f"Debug: Dataloader has {len(dataloader)} batches")
            print(f"Debug: Dataloader batch size: {dataloader.batch_size}")
            
            # Use the same evaluation wrapper that worked during calibration
            def evaluate_with_gradients(model, data):
                # Enable gradients for LLC estimation
                for param in model.parameters():
                    param.requires_grad_(True)
                
                # Fix device mismatch - move data to model's device
                x, y = data
                model_device = next(model.parameters()).device
                x = x.to(model_device)
                y = y.to(model_device)
                
                # Use the original evaluate_ce function
                return evaluate_ce(model, (x, y))
            
            # Debug: Test the evaluation function first
            print("Debug: Testing evaluation function...")
            try:
                test_batch = next(iter(dataloader))
                test_loss, test_info = evaluate_with_gradients(model, test_batch)
                print(f"Debug: Test evaluation successful, loss: {test_loss.item():.6f}")
                print(f"Debug: Test info keys: {list(test_info.keys())}")
            except Exception as e:
                print(f"Debug: Test evaluation failed: {e}")
                import traceback
                traceback.print_exc()
            
            llc_stats = estimate_learning_coeff_with_summary(
                model=model,
                loader=dataloader,
                evaluate=evaluate_with_gradients,  # Use same wrapper as calibration it's effectively evaluate_ce
                sampling_method=SGLD,
                optimizer_kwargs=dict(lr=best_epsilon, localization=gamma, nbeta=nbeta),
                num_chains=3,  # Use same as calibration 
                num_draws=500,  # Increased from 100 to 500 for stability
                num_burnin_steps=450,  # Increased from 90 to 450 (90% of 500)
                device=self.device,
                online=True,
                verbose=True,
            )
            
            # Print results
            # llc_mean = llc_stats.get('llc/mean', np.nan)
            # llc_std = llc_stats.get('llc/std', np.nan)
            llc_mean, llc_std = self._extract_llc_values(llc_stats)

            print(f"Test run result: LLC = {llc_mean:.4f} ± {llc_std:.4f}")
            
            # Debug: Check what's in the stats
            print(f"Debug: Available stats keys: {list(llc_stats.keys())}")
            
            # Debug: Try different ways to extract LLC
            print(f"Debug: llc/mean from stats: {llc_stats.get('llc/mean', 'NOT_FOUND')}")
            print(f"Debug: llc/std from stats: {llc_stats.get('llc/std', 'NOT_FOUND')}")
            
            # Check if there are other possible keys
            # for key in llc_stats.keys():
            #     if 'llc' in key.lower():
            #         print(f"Debug: {key}: {llc_stats[key]}")
            
            if 'llc/trace' in llc_stats:
                trace = llc_stats['llc/trace']
                print(f"Debug: Trace contains NaN: {np.isnan(trace).any()}")
                print(f"Debug: Trace contains Inf: {np.isinf(trace).any()}")
                print(f"Debug: Trace min/max: {trace.min():.6f} / {trace.max():.6f}")
                if np.isnan(trace).any():
                    print(f"Debug: NaN positions: {np.where(np.isnan(trace))}")
                
                # Debug: Check the final calculation
                if 'llc/means' in llc_stats:
                    print(f"Debug: llc/means exists")
                if 'llc/stds' in llc_stats:
                    print(f"Debug: llc/stds exists")
                
                # Manual calculation to see what's happening
                print(f"Debug: Manual trace mean: {trace.mean():.6f}")
                print(f"Debug: Manual trace std: {trace.std():.6f}")
                print(f"Debug: Manual trace std/mean: {trace.std()/trace.mean():.6f}")
                
                # Convergence diagnostics (per devinterp docs)
                if trace.mean() < 0:
                    print("⚠️  WARNING: Negative LLC detected - possible SGLD wandering or poor initialization")
                
                cv = abs(trace.std() / trace.mean()) if trace.mean() != 0 else float('inf')
                if cv > 2.0:
                    print(f"⚠️  WARNING: High coefficient of variation ({cv:.2f}) - chains may not have converged")
                    print("    Consider: longer burn-in, smaller step size, or more chains")
                
                # Check for chain mixing issues
                if trace.max() - trace.min() > 10 * abs(trace.mean()):
                    print("⚠️  WARNING: Extreme trace range detected - possible poor chain mixing")
                    print("    Consider: adjusting gamma, epsilon, or increasing burn-in")
            
            # Plot the trace if available
            if 'llc/trace' in llc_stats:
                try:
                    print("Plotting LLC trace...")
                    
                    # Use the new plot_sampling_evolution function
                    plot_path = Path(f"llc_calibration_results_{checkpoint_name}.png")
                    plot_sampling_evolution(llc_stats, save_path=str(plot_path), show=False)
                    
                except Exception as plot_error:
                    print(f"Trace plotting failed: {plot_error}")
                    import traceback
                    traceback.print_exc()
            

            
            return llc_stats
            
        except Exception as e:
            print(f"Test run failed: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to manual calibration...")
            return None
    
    
    def _estimate_llc_for_calibration(self, model, evaluate, device, loader, 
                                    epsilon, beta):
        """Helper for calibration sweep"""
        try:
            # Use default_nbeta for proper beta calculation
            nbeta = default_nbeta(loader)
            
            # Create a wrapper that ensures gradients are enabled
            def evaluate_with_gradients(model, data):
                # Enable gradients for LLC estimation
                for param in model.parameters():
                    param.requires_grad_(True)
                # Use the original evaluate_ce function
                return evaluate_ce(model, data)
            
            # Use the wrapped evaluation function
            stats = estimate_learning_coeff_with_summary(
                model=model,
                loader=loader,
                evaluate=evaluate_with_gradients,  # Use wrapped evaluate_ce
                sampling_method=SGLD,
                optimizer_kwargs=dict(lr=epsilon, localization=5.0, nbeta=nbeta),  # Increased localization to prevent wandering
                num_chains=4,  # Guidelines: 4-20 chains
                num_draws=100,  # Reduced for faster calibration
                num_burnin_steps=90,  # 90% of 100 total steps
                device=device,
                online=True,
                verbose=False,
            )
            # Return the full stats dictionary as expected by EpsilonBetaAnalyzer
            return stats
        except Exception as e:
            print(f"Error in calibration for epsilon={epsilon:.2e}, beta={beta:.3f}: {e}")
            # Return a dictionary with NaN values to indicate failure
            return {'llc/mean': np.nan, 'llc/std': np.nan}


def plot_sampling_evolution(
    llc_stats: Dict[str, Any], 
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 9)
) -> plt.Figure:
    """Plot LLC loss traces from LLC estimation.
    
    This recreates the original plot_sampling_evolution from utils_outdated.py
    
    Args:
        llc_stats: Dictionary containing 'loss/trace' or 'llc/trace' and optionally 'llc_average_mean'
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Try to get trace from either 'loss/trace' or 'llc/trace'
    if 'loss/trace' in llc_stats:
        trace = llc_stats['loss/trace']
        trace_type = "Loss"
    elif 'llc/trace' in llc_stats:
        trace = llc_stats['llc/trace']
        trace_type = "LLC"
    else:
        raise ValueError("No 'loss/trace' or 'llc/trace' found in llc_stats")
    
    print(f"{trace_type} trace shape:", trace.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get dimensions correctly
    num_chains, num_draws = trace.shape 
    sgld_step = list(range(num_draws))
    
    # Plot individual chains
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(num_chains):
        draws = trace[i]
        color = colors[i % len(colors)]
        ax.plot(sgld_step, draws, linewidth=1.5, label=f"chain {i}", 
                color=color, alpha=0.8)

    # Plot mean with black dashed line
    mean = np.mean(trace, axis=0)
    ax.plot(sgld_step, mean, color="black", linestyle="--", 
            linewidth=3, label="mean", zorder=3)

    # Add std shading
    std = np.std(trace, axis=0)
    ax.fill_between(sgld_step, mean - std, mean + std, 
                    color="gray", alpha=0.3, zorder=2)

    # Formatting
    # Try to get llc_average_mean, fallback to calculated mean if not available
    llc_avg = llc_stats.get('llc_average_mean', np.mean(trace))
    ax.set_title(f"{trace_type} Trace, avg LLC = {llc_avg:.2f}",
                fontsize=16, fontweight='bold')
    ax.set_xlabel("SGLD Step", fontsize=14)
    ax.set_ylabel(trace_type, fontsize=14)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save and show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Trace plot saved to {save_path}")
    if show:
        plt.show()
    
    return fig
    
    