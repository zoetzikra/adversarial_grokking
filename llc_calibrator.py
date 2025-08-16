#!/usr/bin/env python3
"""
LLC Hyperparameter Calibrator

Extracted from the working reference_effective-dim_llc_measurement.py file,
this class provides robust hyperparameter calibration using the devinterp
EpsilonBetaAnalyzer with proven selection criteria.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import copy

# devinterp imports
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import default_nbeta, evaluate_ce
from devinterp.vis_utils import EpsilonBetaAnalyzer
from devinterp.slt.mala import MalaAcceptanceRate


def convert_to_serializable(obj):
    """Convert numpy arrays and scalar types to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


@dataclass
class LLCCalibratorConfig:
    """Configuration for LLC measurement following empirical guide recommendations"""
    
    # Model information (for size-based scaling)
    num_parameters: Optional[int] = None  # For automatic scaling
    
    # SGLD hyperparameters
    epsilon: float = 1e-4  # Step size (learning rate) - will be auto-scaled
    gamma: float = None   # Localization strength (start small per guide)
    nbeta: Optional[float] = None  # Inverse temperature (auto-set if None)
    
    # Sampling parameters (following guide recommendations)
    num_chains: int = 8  # 4-20 chains recommended
    # num_steps: int = 2000  # Total steps per chain (not draws)
    num_steps: int = 500 # so the draws will be 500 - 500*90/100 = 50
    num_burnin_steps: Optional[int] = None  # Auto-set to 90% of total steps
    num_steps_bw_draws: int = 1  # Steps between draws
    
    # Batch size for SGLD
    batch_size: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Calibration parameters (expanded ranges per guide)
    calibration_epsilons: List[float] = None
    
    # SGLD parameters for calibration runs
    # gamma: float = 10.0  # Default localization strength
    batch_size: int = 512
    
    # Diagnostic targets
    target_mala_acceptance: float = 0.92  # Target MALA acceptance rate (0.9-0.95)
    
    def __post_init__(self):
        # Set default calibration ranges based on guide recommendations
        if self.calibration_epsilons is None:
            # Expanded range for better coverage
            self.calibration_epsilons = [1e-6, 1e-5, 1e-4, 1e-3]
        
        # Auto-set burn-in to 90% of total steps (guide recommendation)
        if self.num_burnin_steps is None:
            self.num_burnin_steps = int(0.9 * self.num_steps)
        
        # Auto-scale epsilon based on model size if num_parameters provided
        if self.num_parameters is not None:
            self.epsilon = self._scale_epsilon_by_model_size()
        
        # Auto-scale gamma based on model type
        # self.gamma = 50.0
        # self.gamma = 2000.0
        # self.gamma = 10000.0
        self.gamma = 5000.0
        # self.gamma = 1000.0
        

    def _scale_epsilon_by_model_size(self) -> float:
        """Scale epsilon based on model size per guide recommendations"""
        if self.num_parameters is None:
            return self.epsilon
        
        # Guide recommendations:
        # Small networks (≤10K params): ε = 1e-3 to 1e-5
        # Medium networks (100K-1M params): ε = 1e-4 to 1e-6
        # Large networks (>1M params): ε = 1e-5 to 1e-7
        
        if self.num_parameters <= 10_000:
            return 1e-4  # Small networks
        elif self.num_parameters <= 1_000_000:
            return 1e-5  # Medium networks
        else:
            return 1e-6  # Large networks
    
    
    def get_effective_draws(self) -> int:
        """Calculate effective number of draws after burn-in"""
        return self.num_steps - self.num_burnin_steps
    
    def validate_for_model(self, model: nn.Module) -> None:
        """Validate configuration for a specific model"""
        if self.num_parameters is None:
            # Count parameters
            self.num_parameters = sum(p.numel() for p in model.parameters())
            print(f"Model has {self.num_parameters:,} parameters")
            
            # Re-scale based on actual model size
            self.epsilon = self._scale_epsilon_by_model_size()
            print(f"Auto-scaled epsilon to {self.epsilon:.2e}")
        
        # Print configuration summary
        print(f"\n=== LLC Configuration Summary ===")
        print(f"Model parameters: {self.num_parameters:,}")
        print(f"Step size (ε): {self.epsilon:.2e}")
        print(f"Localization (γ): {self.gamma:.1f}")
        print(f"Chains: {self.num_chains}")
        print(f"Total steps: {self.num_steps}")
        print(f"Burn-in steps: {self.num_burnin_steps} ({100*self.num_burnin_steps/self.num_steps:.0f}%)")
        print(f"Effective draws: {self.get_effective_draws()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Target MALA acceptance: {self.target_mala_acceptance:.2f}")
        print("=" * 40)


class LLCCalibrator:
    """
    LLC Hyperparameter Calibrator using EpsilonBetaAnalyzer.
    
    This class extracts the proven calibration methodology from the reference
    file and provides a clean interface for hyperparameter optimization.
    """
    
    def __init__(self, config: LLCCalibratorConfig = None):
        self.config = config or LLCCalibratorConfig()
        self.device = self.config.device
        
    def evaluate_model(self, model: nn.Module, data: tuple) -> tuple:
        """
        Clean evaluation function for LLC estimation.
        Simplified from reference file - focuses on clean data only.
        """
        # Keep model in eval mode but ensure gradients are enabled
        model.eval()
        
        # CRITICAL: Ensure all parameters require gradients for LLC estimation
        for param in model.parameters():
            param.requires_grad_(True)
        
        inputs, targets = data
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Ensure model is on correct device
        model_device = next(model.parameters()).device
        target_device = torch.device(self.device)
        
        if model_device != target_device:
            model = model.to(target_device)
        
        # Forward pass
        outputs = model(inputs)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("⚠️  NaN/Inf detected in model outputs")
            return torch.tensor(float('inf'), device=self.device), {}
        
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  NaN/Inf detected in loss")
            return torch.tensor(float('inf'), device=self.device), {}
        

        return loss, {"logits": outputs}
    
    def calibrate_hyperparameters_multi_gamma(self,
                                         model: nn.Module,
                                         train_loader: DataLoader,
                                         gamma_values: List[float],
                                         checkpoint_name: str = None,
                                         save_path: str = None,
                                         use_tuned_beta: bool = True) -> Dict[str, float]:
        """
        Calibrate SGLD hyperparameters across multiple gamma values, selecting the best
        (epsilon, gamma, beta) combination based on MALA acceptance rate.
        
        Args:
            model: Model to calibrate for
            train_loader: Training data loader
            gamma_values: List of gamma values to test
            checkpoint_name: Name for saving results
            save_path: Directory to save calibration results
            use_tuned_beta: Whether to use tuned beta from analyzer or default
            
        Returns:
            Dictionary with optimal hyperparameters selected based on MALA acceptance
        """
        print("============================================================")
        print("STARTING MULTI-GAMMA LLC HYPERPARAMETER CALIBRATION")
        print("============================================================")
        
        all_results = []
        original_gamma = self.config.gamma
        
        try:
            for i, gamma in enumerate(gamma_values):
                print(f"\n{'='*60}")
                print(f"GAMMA SWEEP {i+1}/{len(gamma_values)}: γ = {gamma}")
                print(f"{'='*60}")
                
                # Create a deep copy of the model for each gamma to prevent contamination
                model_copy = copy.deepcopy(model)
                print(f"  Created fresh model copy with {sum(p.numel() for p in model_copy.parameters()):,} parameters")
                
                # Temporarily set this gamma value
                self.config.gamma = gamma
                
                # Run calibration for this gamma with the fresh model copy
                try:
                    result = self.calibrate_hyperparameters(
                        model=model_copy,  # Use fresh copy instead of contaminated original
                        train_loader=train_loader,
                        checkpoint_name=f"{checkpoint_name}_gamma_{gamma}" if checkpoint_name else None,
                        save_path=f"{save_path}/gamma_{gamma}" if save_path else None,
                        selection_method="mala",  # Always use MALA for multi-gamma
                        use_tuned_beta=use_tuned_beta
                    )
                    
                    # Add gamma to the result
                    result['gamma'] = gamma
                    all_results.append(result)
                    
                    print(f"✅ Gamma {gamma} completed successfully")
                    print(f"   Selected: ε={result['epsilon']:.2e}, β={result['nbeta']:.3f}")
                    if 'mala_acceptance' in result:
                        print(f"   MALA acceptance: {result['mala_acceptance']:.3f}")
                    
                except Exception as e:
                    print(f"❌ Gamma {gamma} failed: {e}")
                    continue
            
            # Select best result based on MALA acceptance rate
            return self._select_best_multi_gamma_result(all_results)
            
        finally:
            # Restore original gamma
            self.config.gamma = original_gamma

    def calibrate_hyperparameters(self, 
                                model: nn.Module, 
                                train_loader: DataLoader,
                                checkpoint_name: str = None,
                                save_path: str = None,
                                selection_method: str = "mala",
                                use_tuned_beta: bool = False) -> Dict[str, float]:
        """
        Calibrate SGLD hyperparameters using EpsilonBetaAnalyzer.
        
        Args:
            model: Model to calibrate for
            train_loader: Training data loader
            checkpoint_name: Name for saving results
            save_path: Path to save calibration results
            selection_method: "stability" (default) or "mala" for parameter selection
            use_tuned_beta: If True, use tuned beta; if False, use default_nbeta
            
        Returns:
            Dictionary with optimal hyperparameters
        """
        print("=" * 60)
        print("STARTING LLC HYPERPARAMETER CALIBRATION")
        print("=" * 60)
        
        # Setup save path
        if save_path is None and checkpoint_name is not None:
            save_path = f"./llc_calibration_results/{checkpoint_name}"
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"Results will be saved to: {save_path}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params:,} parameters")
        
        #TODO: maybe use a sample of the datasetfor calibration (that's how it is in the calibration script)
        # Create calibration data loader
        calibration_loader = DataLoader(
            train_loader.dataset, # OPTIONAL [:1000]
            batch_size=self.config.batch_size,
            shuffle=True
        )
       
        
        print(f"Using {len(train_loader.dataset)} samples for calibration")
        print(f"Testing {len(self.config.calibration_epsilons)} epsilon values")
        print(f"Epsilon range: {[f'{e:.1e}' for e in self.config.calibration_epsilons]}")
        print(f"Gamma value tested: {self.config.gamma}")
        
        # Initialize EpsilonBetaAnalyzer (the proven devinterp tool)
        analyzer = EpsilonBetaAnalyzer()
        
        # Configure the sweep
        analyzer.configure_sweep(
            llc_estimator=self._estimate_llc_for_calibration,
            llc_estimator_kwargs=dict(
                model=model,
                evaluate=self.evaluate_model,
                device=self.device,
                loader=calibration_loader,
            ),
            min_epsilon=min(self.config.calibration_epsilons),
            max_epsilon=max(self.config.calibration_epsilons),
            epsilon_samples=len(self.config.calibration_epsilons),
            min_beta=None,  # Let analyzer determine beta range
            max_beta=None,
            beta_samples=3,  # Good coverage of beta space
            dataloader=calibration_loader,
        )
        
        # Run the sweep
        print("\nRunning epsilon-beta sweep with EpsilonBetaAnalyzer...")
        analyzer.sweep()
        
        # Note: analyzer.plot() calls are broken, so we skip plotting the sweep results
        # The important part is the parameter selection, not the plots
        
        # Extract optimal parameters using chosen selection criteria
        optimal_params = self._extract_optimal_params(analyzer, calibration_loader, selection_method, use_tuned_beta)
        
       
        # Save calibration results
        if save_path:
            calibration_results = {
                "optimal_params": optimal_params,
                "calibration_epsilons": self.config.calibration_epsilons,
                "model_parameters": num_params,
                "checkpoint_name": checkpoint_name,
            }
            
            # Convert to serializable format before saving
            serializable_results = convert_to_serializable(calibration_results)
            with open(os.path.join(save_path, "calibration_results.json"), "w") as f:
                json.dump(serializable_results, f, indent=2)
        
        print("=" * 60)
        print("CALIBRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"✅ Optimal Epsilon: {optimal_params['epsilon']:.2e}")
        print(f"✅ Optimal Gamma: {optimal_params['gamma']:.3f}")
        print(f"✅ Optimal Beta: {optimal_params['nbeta']:.3f}")
        if 'llc_mean' in optimal_params:
            print(f"✅ Expected LLC: {optimal_params['llc_mean']:.4f}")
        print("=" * 60)
        
        return optimal_params
    
    def _estimate_llc_for_calibration(self, 
                                    model: nn.Module,
                                    loader: DataLoader,
                                    evaluate: callable,
                                    epsilon: float,
                                    beta: float,
                                    **kwargs) -> Dict:
        """Helper function for calibration sweep - used by EpsilonBetaAnalyzer"""
        
        # Create MALA callback for calibration runs too
        mala_callback = MalaAcceptanceRate(
            num_chains=2,  # Match the number of chains used in calibration
            num_draws=10,  # Match the number of draws used in calibration
            nbeta=beta,
            learning_rate=epsilon,
            device=self.device
        )
        
        return estimate_learning_coeff_with_summary(
            model=model,
            loader=loader,
            evaluate=evaluate,
            sampling_method=SGLD,
            optimizer_kwargs=dict(
                lr=epsilon, 
                localization=self.config.gamma,  # Use fixed gamma for calibration
                nbeta=beta
            ),
            num_chains=2,  # Use fewer chains for calibration speed
            num_draws=10,  # Use fewer draws for calibration speed  # DRAWS NOT STEPS
            num_burnin_steps=90,  # 90% burn-in for calibration
            device=self.device,
            online=True,
            callbacks=[mala_callback],  # Add MALA callback for acceptance rate tracking
        )
    
    def _extract_optimal_params(self, analyzer: EpsilonBetaAnalyzer, dataloader: DataLoader, selection_method: str = "stability", use_tuned_beta: bool = False) -> Dict[str, float]:
        """
        Extract optimal parameters using the proven selection criteria from reference file.
        This is the sophisticated logic that actually works!
        """
        print("\n=== HYPERPARAMETER SELECTION ===")
        
        if analyzer.sweep_df is None or len(analyzer.sweep_df) == 0:
            print("No calibration results found, using default parameters")
            return self._get_default_params(dataloader)
        
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
        
        if len(valid_results) == 0:
            print("No valid results found, using default parameters")
            return self._get_default_params(dataloader)
        
        # Debug: Check for negative LLC values
        print(f"LLC value range: {valid_results['llc/final'].min():.4f} to {valid_results['llc/final'].max():.4f}")
        print(f"Stability range: {valid_results['llc/std_over_mean'].min():.4f} to {valid_results['llc/std_over_mean'].max():.4f}")
        
        # Check for negative LLC values (failure mode according to devinterp guide)
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
        
        # Debug: Check what columns are available
        print(f"Available result columns: {list(clean_results.columns)}")
        
        # Parameter selection based on chosen method
        if selection_method == "mala":
            print("Using MALA acceptance rate-based parameter selection")
            # Check if MALA data is actually available
            if 'mala_accept/mean' in clean_results.columns:
                best_row = self._select_by_mala_acceptance(clean_results, positive_llc)
            else:
                print("⚠️  MALA selection requested but no MALA data found")
                print("   This likely means EpsilonBetaAnalyzer doesn't preserve callback results")
                print("   Falling back to stability-based selection")
                best_row = self._select_by_stability(clean_results, positive_llc)
        else:  # Default: stability-based
            print("Using stability-based parameter selection")
            best_row = self._select_by_stability(clean_results, positive_llc)
        
        # Choose beta based on use_tuned_beta flag
        if use_tuned_beta and 'beta' in best_row:
            selected_beta = float(best_row['beta'])
            beta_source = "tuned"
        else:
            selected_beta = float(default_nbeta(dataloader))
            beta_source = "default"
        
        # Get llc/stds from results if available (treat like llc/means - take mean of the array)
        if 'llc/stds' in best_row and best_row['llc/stds'] is not None:
            llc_std = float(best_row['llc/stds'].mean())
        else:
            llc_std = None
        
        # Extract MALA acceptance rate if available
        mala_acceptance = None
        if 'mala_accept/mean' in best_row:
            mala_acceptance = float(best_row['mala_accept/mean'])
        
        best_params = {
            'epsilon': float(best_row['epsilon']),
            'gamma': float(self.config.gamma),  # Always from config (not tuned)
            'nbeta': selected_beta,  
            'llc_mean': float(best_row['llc/final']), 
            'llc_std': llc_std,
            'llc_std_over_mean': float(best_row['llc/std_over_mean']),
            'beta_source': beta_source,
            'gamma_source': 'fixed_config',  # Make it clear gamma is not tuned
            'mala_acceptance': mala_acceptance  # Include MALA acceptance rate
        }
        
        print(f"\nSelected parameters ({selection_method}-based selection):")
        print(f"  ε = {best_params['epsilon']:.2e} (tuned via EpsilonBetaAnalyzer)")
        print(f"  γ = {best_params['gamma']:.3f} (FIXED - not tuned)")
        print(f"  β = {best_params['nbeta']:.3f} ({beta_source})")
        print(f"  LLC mean = {best_params['llc_mean']:.4f}")
        print(f"  LLC std/mean = {best_params['llc_std_over_mean']:.4f}")
        if 'llc_std' in best_params and best_params['llc_std'] is not None:
            print(f"  LLC std = {best_params['llc_std']:.4f}")
        if best_params['mala_acceptance'] is not None:
            print(f"  MALA acceptance = {best_params['mala_acceptance']:.3f}")
        
        if best_params['llc_mean'] < 0:
            print(f"  ⚠️  WARNING: Negative LLC selected - may need parameter adjustment")
        
        # Also print top 5 results for comparison
        print(f"\nTop 5 parameter combinations (by stability):")
        clean_results['abs_stability'] = clean_results['llc/std_over_mean'].abs()
        top_5 = clean_results.nsmallest(5, 'abs_stability')
        for i, (_, row) in enumerate(top_5.iterrows()):
            llc_sign = "⚠️" if row['llc/final'] < 0 else "✓"
            print(f"  {i+1}. {llc_sign} ε={row['epsilon']:.2e}, LLC={row['llc/final']:.4f}, stability={row['llc/std_over_mean']:.4f}")
        
        print("=" * 50)
        return best_params
    
    def _select_by_stability(self, clean_results, positive_llc):
        """Select parameters based on stability (std/mean ratio)"""
        # Prefer positive LLC with good stability
        if len(positive_llc) > 0:
            # Find the most stable positive LLC result
            best_stability_idx = positive_llc['llc/std_over_mean'].abs().idxmin()
            best_row = positive_llc.loc[best_stability_idx]
            print(f"Best row: {best_row}")
            print("✓ Using positive LLC result with best stability")
        else:
            # Fall back to best stability among all results
            best_stability_idx = clean_results['llc/std_over_mean'].abs().idxmin()
            best_row = clean_results.loc[best_stability_idx]
            print("⚠️  No positive LLC found, using best stability among all results")
        return best_row
    
    def _select_by_mala_acceptance(self, clean_results, positive_llc):
        """Select parameters based on MALA acceptance rate (assumes data is available)"""
        print(f"✓ MALA acceptance data found in calibration results")
        mala_rates = clean_results['mala_accept/mean']
        print(f"  MALA acceptance range: {mala_rates.min():.3f} - {mala_rates.max():.3f}")
        
        # Target acceptance rate: 0.90-0.95 is ideal
        target_acceptance = 0.92
        
        # Calculate distance from target acceptance rate
        clean_results['mala_distance'] = (clean_results['mala_accept/mean'] - target_acceptance).abs()
        
        # Prefer positive LLC results
        if len(positive_llc) > 0:
            # Update positive_llc to include the new mala_distance column
            positive_llc_with_distance = clean_results[clean_results['llc/final'] > 0].copy()
            # Among positive LLC, find closest to target MALA acceptance
            best_mala_idx = positive_llc_with_distance['mala_distance'].idxmin()
            best_row = positive_llc_with_distance.loc[best_mala_idx]
            acceptance_rate = best_row['mala_accept/mean']
            print(f"✓ Selected positive LLC with MALA acceptance = {acceptance_rate:.3f} (target: {target_acceptance:.3f})")
        else:
            # Fall back to best MALA acceptance among all results
            best_mala_idx = clean_results['mala_distance'].idxmin()
            best_row = clean_results.loc[best_mala_idx]
            acceptance_rate = best_row['mala_accept/mean']
            print(f"⚠️  No positive LLC found, using best MALA acceptance = {acceptance_rate:.3f}")
        
        return best_row
    
    def _select_best_multi_gamma_result(self, all_results: List[Dict]) -> Dict[str, float]:
        """
        Select the best result from multiple gamma calibrations based on MALA acceptance rate.
        """
        if not all_results:
            raise ValueError("No successful calibration results to choose from")
        
        print(f"\n{'='*60}")
        print("MULTI-GAMMA RESULT COMPARISON")
        print(f"{'='*60}")
        
        # Target MALA acceptance rate (0.90-0.95 is ideal)
        target_acceptance = 0.92
        
        # Find results with MALA data and calculate distances from target
        valid_results = []
        for result in all_results:
            if 'mala_acceptance' in result and result['mala_acceptance'] is not None:
                distance = abs(result['mala_acceptance'] - target_acceptance)
                result['mala_distance'] = distance
                valid_results.append(result)
        
        if not valid_results:
            print("⚠️  No results with MALA data found, falling back to first successful result")
            return all_results[0]
        
        # Sort by MALA acceptance distance (best first)
        valid_results.sort(key=lambda x: x['mala_distance'])
        
        # Display comparison
        print(f"Results ranked by MALA acceptance (target = {target_acceptance:.3f}):")
        print("-" * 80)
        for i, result in enumerate(valid_results):
            gamma = result['gamma']
            epsilon = result['epsilon']
            beta = result['nbeta']
            mala_acc = result['mala_acceptance']
            llc_mean = result.get('llc_mean', 'N/A')
            llc_std = result.get('llc_std_over_mean', 'N/A')
            
            marker = "🏆" if i == 0 else f" {i+1}."
            print(f"{marker} γ={gamma:>6.0f}, ε={epsilon:.2e}, β={beta:>6.1f}, "
                  f"MALA={mala_acc:.3f} (Δ={result['mala_distance']:.3f}), "
                  f"LLC={llc_mean:.3f}±{llc_std:.3f}")
        
        best_result = valid_results[0]
        print(f"\n🏆 SELECTED BEST COMBINATION:")
        print(f"   γ = {best_result['gamma']}")
        print(f"   ε = {best_result['epsilon']:.2e}")
        print(f"   β = {best_result['nbeta']:.3f}")
        print(f"   MALA acceptance = {best_result['mala_acceptance']:.3f}")
        print(f"   LLC mean = {best_result.get('llc_mean', 'N/A')}")
        print(f"{'='*60}")
        
        return best_result
    
    def _get_default_params(self, dataloader: DataLoader) -> Dict[str, float]:
        """Get default parameters when calibration fails"""
        return {
            "epsilon": float(np.mean(self.config.calibration_epsilons)),
            "gamma": float(self.config.gamma),
            "nbeta": float(default_nbeta(dataloader))
        }
    
    def load_calibrated_hyperparameters(self, calibration_path: str) -> Optional[Dict[str, float]]:
        """
        Load pre-calibrated hyperparameters from a previous calibration run.
        """
        try:
            print(f"Loading calibration from: {calibration_path}")
            
            if not os.path.exists(calibration_path):
                print(f"File does not exist: {calibration_path}")
                return None
            
            with open(calibration_path, 'r') as f:
                calibration_results = json.load(f)
            
            if 'optimal_params' in calibration_results:
                optimal_params = calibration_results['optimal_params']
                print(f"✅ Successfully loaded pre-calibrated hyperparameters")
                print(f"  ε = {optimal_params['epsilon']:.2e}")
                print(f"  γ = {optimal_params['gamma']:.3f}")
                print(f"  β = {optimal_params['nbeta']:.3f}")
                if 'llc_mean' in optimal_params:
                    print(f"  Previous LLC = {optimal_params['llc_mean']:.4f}")
                return optimal_params
            else:
                print("No 'optimal_params' found in calibration results")
                return None
                
        except Exception as e:
            print(f"Error loading calibrated hyperparameters: {e}")
            return None


    def estimate_llc(self, 
                    model: nn.Module, 
                    train_loader: DataLoader,
                    hyperparams: Optional[Dict[str, float]] = None,
                    save_path: Optional[str] = None,
                    seed: Optional[int] = None,
        ) -> Dict[str, Any]:
        """
        Estimate LLC for a single model.
        
        Args:
            model: Model to estimate LLC for
            train_loader: Training data loader
            hyperparams: Optional hyperparameters (uses config defaults if None)
            save_path: Optional path to save diagnostic plots
            seed: Optional seed for reproducible sampling
            
        Returns:
            Dictionary containing LLC estimates and diagnostics
        """
        # Set seed for reproducible sampling if provided
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            import numpy as np
            import random
            np.random.seed(seed)
            random.seed(seed)
        
        # Validate configuration for this model
        self.config.validate_for_model(model)
        
        if hyperparams is None:
            hyperparams = {
                "epsilon": self.config.epsilon,
                "gamma": self.config.gamma,
                "nbeta": self.config.nbeta or default_nbeta(train_loader)
            }
        
        print(f"Estimating LLC with hyperparams: {hyperparams}")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Target device: {self.device}")
        
        # Move model to correct device if needed (only once)
        model_device = next(model.parameters()).device
        target_device = torch.device(self.device)
        
        if model_device != target_device:
            print(f"  Moving model to {self.device}")
            model = model.to(target_device)
        
        #TODO: TRY THIS NEXT
        # Keep model in eval mode but ensure gradients are enabled
        model.eval()
        
        # CRITICAL: Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad_(True)
        
        # Verify gradient setup
        grad_params = [p for p in model.parameters() if p.requires_grad]
        non_grad_params = [p for p in model.parameters() if not p.requires_grad]
        print(f"  Parameters requiring gradients: {len(grad_params)}")
        print(f"  Parameters NOT requiring gradients: {len(non_grad_params)}")
        
        if len(grad_params) == 0:
            raise ValueError("No model parameters require gradients!")
        
        # Debug: Print first few parameter names and their grad status
        if len(non_grad_params) > 0:
            print("  ⚠️  Some parameters don't require gradients:")
            for i, (name, param) in enumerate(model.named_parameters()):
                if not param.requires_grad:
                    print(f"    {name}: requires_grad = {param.requires_grad}")
                if i >= 5:  # Show only first 5
                    break
        
        # Prepare data loader
        llc_loader = DataLoader(
            train_loader.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"⚠️  NaN in {name}")
                raise ValueError("Model contains NaN parameters")
            if torch.isinf(param).any():
                print(f"⚠️  Inf in {name}")
                raise ValueError("Model contains Inf parameters")
    

        # Create MALA acceptance rate callback
        mala_callback = MalaAcceptanceRate(
            num_chains=self.config.num_chains,
            num_draws=self.config.get_effective_draws(),
            nbeta=hyperparams["nbeta"],  # Use nbeta directly (no conversion needed)
            learning_rate=hyperparams["epsilon"],
            device=self.device
        )

    
        try:
            # Run LLC estimation
            results = estimate_learning_coeff_with_summary(
                model=model,
                loader=llc_loader,
                evaluate=self.evaluate_model,
                sampling_method=SGLD,
                optimizer_kwargs=dict(
                    lr=hyperparams["epsilon"],
                    localization=hyperparams["gamma"],
                    nbeta=hyperparams["nbeta"],
                ),
                num_chains=self.config.num_chains,
                num_draws=self.config.get_effective_draws(),  # Use effective draws
                num_burnin_steps=self.config.num_burnin_steps,
                num_steps_bw_draws=self.config.num_steps_bw_draws,
                device=self.device,
                online=True,
                callbacks=[mala_callback],
            )
            
            print(f"  LLC estimation completed successfully")
            print(f"  Results keys: {list(results.keys())}")
            
            # Add llc/mean for compatibility (mean across all chains)
            # Note: llc/stds is already provided by devinterp
            if 'llc/means' in results:
                results['llc/mean'] = float(results['llc/means'].mean())
        

            # Access MALA acceptance rate results
            if 'mala_accept/mean' in results:
                acceptance_rate = results['mala_accept/mean']
                print(f"  Average MALA acceptance rate: {acceptance_rate:.3f}")
                
                # Check if acceptance rate indicates numerical problems
                if acceptance_rate < 0.5:
                    print("  ⚠️  Low acceptance rate - potential numerical instability")
                    print("     Consider: increasing gamma, reducing epsilon")
                elif acceptance_rate > 0.98:
                    print("  ✓ Very high acceptance rate - can potentially reduce gamma for efficiency")
                elif 0.9 <= acceptance_rate <= 0.95:
                    print("  ✓ Excellent acceptance rate - parameters well-tuned")
            
            
            return results
            
        except Exception as e:
            print(f"  ERROR in LLC estimation: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            print(f"  Full traceback:")
            traceback.print_exc()
            raise
    



    def plot_sampling_evolution(self,
        llc_stats: Dict[str, Any], 
        save_path: Optional[str] = None,
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
        else:
            plt.close(fig)  # Close figure if not showing to save memory
        
        return fig
        
        

    '''
    NOT IN USE CURRENTLY

    In general about MALA acceptance rate:
    Too low → reduce epsilon
    Too high → increase epsilon
    '''
    def auto_adjust_hyperparameters(self, hyperparams, acceptance_rate):
        """Auto-adjust hyperparameters based on MALA acceptance rate"""
        
        # PRIMARY: Adjust epsilon based on MALA acceptance rate
        if acceptance_rate < 0.5:
            # Step size too large - reduce epsilon
            new_epsilon = hyperparams["epsilon"] * 0.8
            print(f"⚠️  Auto-reducing epsilon: {hyperparams['epsilon']:.2e} → {new_epsilon:.2e}")
            hyperparams["epsilon"] = new_epsilon
            
        elif acceptance_rate > 0.98:
            # Step size too small - increase epsilon
            new_epsilon = hyperparams["epsilon"] * 1.2  
            print(f"✓ Auto-increasing epsilon: {hyperparams['epsilon']:.2e} → {new_epsilon:.2e}")
            hyperparams["epsilon"] = new_epsilon
            
        elif 0.9 <= acceptance_rate <= 0.95:
            print("✓ Excellent MALA acceptance rate - epsilon well-tuned")
        
        # SECONDARY: Only adjust gamma if you have stability issues
        # (This would be based on NaN detection, not MALA rate)
        
        return hyperparams