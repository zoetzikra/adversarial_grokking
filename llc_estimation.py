import torch
import numpy as np
from torch.utils.data import DataLoader
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce, default_nbeta
from devinterp.vis_utils import EpsilonBetaAnalyzer
import matplotlib.pyplot as plt
import signal
import time

class LLCEstimator:
    def __init__(self, config):
        self.device = config.device
        self.epsilon = config.llc_epsilon  # SGLD step size
        self.gamma = config.llc_gamma      # localization parameter 
        self.num_chains = config.llc_num_chains
        self.num_draws = config.llc_num_draws
        self.num_burnin = config.llc_num_burnin
        
    def estimate_llc(self, model, dataloader, verbose=False):
        """Estimate LLC for current model state"""
        try:
            print("THIS IS CALLED NOW, THE ESTIMATE_LLC FUNCTION")
            print(f"Estimating LLC with: ε={self.epsilon:.2e}, nβ={2.0:.3f}, γ={self.gamma:.2f}")
            stats = estimate_learning_coeff_with_summary(
                model=model,
                loader=dataloader,
                evaluate=evaluate_ce,
                sampling_method=SGLD,
                optimizer_kwargs=dict(
                    lr=self.epsilon, 
                    localization=self.gamma,
                    nbeta=2.0  # Based on notebook calibration
                ),
                num_chains=self.num_chains,
                num_draws=self.num_draws,
                num_burnin_steps=self.num_burnin,
                device=self.device,
                online=True,
                verbose=verbose
            )
            return stats['llc/mean'], stats['llc/std']
        except Exception as e:
            print(f"LLC estimation failed: {e}")
            return np.nan, np.nan
    
    def calibrate_hyperparameters(self, model, dataloader):
        """Calibrate hyperparameters using MALA acceptance probability as diagnostic"""
        print("Starting hyperparameter calibration...")
        print("Following guidelines: Target 0.9-0.95 MALA acceptance probability")
        
        # Get proper nbeta value
        nbeta = default_nbeta(dataloader)
        print(f"Using nbeta = {nbeta:.3f} (from dataloader)")
        
        # Start with conservative parameters based on guidelines
        # For large networks (>1M params), guidelines suggest ε = 1e-5 to 1e-7
        # ResNet18 has ~11M parameters, so start very small
        epsilon_range = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
        gamma_range = [1.0, 5.0, 10.0, 20.0, 50.0]  # Guidelines: 1.0 to 200.0
        
        print(f"Testing epsilon range: {[f'{e:.1e}' for e in epsilon_range]}")
        print(f"Testing gamma range: {gamma_range}")
        
        results = []
        successful_runs = 0
        total_runs = len(epsilon_range) * len(gamma_range)
        
        for epsilon in epsilon_range:
            for gamma in gamma_range:
                print(f"\nTesting ε={epsilon:.1e}, γ={gamma:.1f}")
                
                try:
                    # Use shorter chains for calibration (guidelines: 200-5000 steps)
                    stats = estimate_learning_coeff_with_summary(
                        model=model,
                        loader=dataloader,
                        evaluate=evaluate_ce,
                        sampling_method=SGLD,
                        optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta),
                        num_chains=2,  # Fewer chains for speed
                        num_draws=200,  # Shorter chains for calibration
                        num_burnin_steps=180,  # 90% burn-in
                        device=self.device,
                        online=True,
                        verbose=False,
                    )
                    
                    llc_mean = stats.get('llc/mean', np.nan)
                    llc_std = stats.get('llc/std', np.nan)
                    
                    # Check for MALA acceptance probability if available
                    mala_acceptance = stats.get('mala/acceptance_rate', np.nan)
                    
                    if not np.isnan(llc_mean):
                        results.append({
                            'epsilon': epsilon,
                            'gamma': gamma,
                            'nbeta': nbeta,
                            'llc_mean': llc_mean,
                            'llc_std': llc_std,
                            'mala_acceptance': mala_acceptance,
                            'stability': llc_std / abs(llc_mean) if llc_mean != 0 else np.inf
                        })
                        successful_runs += 1
                        
                        # Print results with acceptance probability
                        if not np.isnan(mala_acceptance):
                            print(f"  ✓ LLC = {llc_mean:.4f} ± {llc_std:.4f}, MALA acceptance = {mala_acceptance:.3f}")
                        else:
                            print(f"  ✓ LLC = {llc_mean:.4f} ± {llc_std:.4f}")
                    else:
                        print(f"  ✗ NaN result")
                        
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        # Print summary
        print(f"\n=== CALIBRATION RESULTS ===")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        
        if not results:
            print("No successful runs found. Model may be too converged for stable LLC estimation.")
            return None
        
        # Sort by MALA acceptance probability (target: 0.9-0.95)
        # If MALA acceptance not available, sort by stability
        if any(not np.isnan(r['mala_acceptance']) for r in results):
            # Use MALA acceptance as primary criterion
            results.sort(key=lambda x: abs(x['mala_acceptance'] - 0.925) if not np.isnan(x['mala_acceptance']) else np.inf)
            print("\n=== PARAMETER SELECTION (MALA Acceptance) ===")
            print("Target: 0.9-0.95 MALA acceptance probability")
            
            for i, result in enumerate(results[:5]):
                mala_acc = result['mala_acceptance']
                if not np.isnan(mala_acc):
                    print(f"  {i+1}. ε={result['epsilon']:.1e}, γ={result['gamma']:.1f}")
                    print(f"     LLC = {result['llc_mean']:.4f} ± {result['llc_std']:.4f}")
                    print(f"     MALA acceptance = {mala_acc:.3f}")
                else:
                    print(f"  {i+1}. ε={result['epsilon']:.1e}, γ={result['gamma']:.1f}")
                    print(f"     LLC = {result['llc_mean']:.4f} ± {result['llc_std']:.4f}")
                    print(f"     MALA acceptance = N/A")
        else:
            # Fallback to stability-based selection
            results.sort(key=lambda x: x['stability'])
            print("\n=== PARAMETER SELECTION (Stability) ===")
            print("MALA acceptance not available, using stability metric")
            
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. ε={result['epsilon']:.1e}, γ={result['gamma']:.1f}")
                print(f"     LLC = {result['llc_mean']:.4f} ± {result['llc_std']:.4f}")
                print(f"     Stability = {result['stability']:.3f}")
        
        # Save detailed results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('llc_calibration_detailed_results.csv', index=False)
        print(f"\nDetailed results saved to: llc_calibration_detailed_results.csv")
        
        # Test the best parameters
        best_params = results[0]
        print(f"\nBest parameters: ε={best_params['epsilon']:.1e}, γ={best_params['gamma']:.1f}")
        
        # Test with longer chains
        test_stats = self._test_best_parameters(model, dataloader, best_params)
        
        return best_params
    
    def _test_best_parameters(self, model, dataloader, best_params):
        """Test the best parameters with a single run and plot trace"""
        print("\n=== TESTING BEST PARAMETERS ===")
        
        # Debug: Print the raw best_params
        print(f"Raw best_params: {best_params}")
        
        # Extract parameters from the new calibration format
        if isinstance(best_params, dict):
            epsilon = best_params.get('epsilon', 1e-8)
            gamma = best_params.get('gamma', 1.0)
            nbeta = best_params.get('nbeta', default_nbeta(dataloader))
        else:
            # Assume it's a tuple or list
            epsilon = best_params[0] if len(best_params) > 0 else 1e-8
            gamma = best_params[1] if len(best_params) > 1 else 1.0
            nbeta = best_params[2] if len(best_params) > 2 else default_nbeta(dataloader)
        
        print(f"Extracted parameters:")
        print(f"  epsilon: {epsilon:.2e}")
        print(f"  gamma: {gamma:.3f}")
        print(f"  nbeta: {nbeta:.3f}")
        print(f"Testing with: ε={epsilon:.2e}, γ={gamma:.3f}, nβ={nbeta:.3f}")
        
        try:
            # Make a single test run with chosen hyperparameters
            print("Making single test run with best hyperparameters...")
            llc_stats = estimate_learning_coeff_with_summary(
                model=model,
                loader=dataloader,
                evaluate=evaluate_ce,
                sampling_method=SGLD,
                optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta),
                num_chains=3,
                num_draws=500,
                num_burnin_steps=450,
                device=self.device,
                online=True,
                verbose=True,
            )
            
            # Print results
            llc_mean = llc_stats.get('llc/mean', np.nan)
            llc_std = llc_stats.get('llc/std', np.nan)
            mala_acceptance = llc_stats.get('mala/acceptance_rate', np.nan)
            
            print(f"Test run result: LLC = {llc_mean:.4f} ± {llc_std:.4f}")
            if not np.isnan(mala_acceptance):
                print(f"MALA acceptance rate: {mala_acceptance:.3f}")
            
            # Plot the trace if available
            if 'llc/trace' in llc_stats:
                try:
                    print("Plotting LLC trace...")
                    
                    # Use custom plotting function instead of devinterp's plot_trace
                    def plot_sampling_evolution(stats, save_path=None, show=False):
                        """Plot LLC loss traces from LLC estimation."""
                        import matplotlib.pyplot as plt
                        
                        trace = stats['llc/trace']
                        print("Loss trace shape:", trace.shape)
                        
                        # Create figure first
                        plt.figure(figsize=(12, 9))
                        
                        # Get dimensions correctly
                        num_chains, num_draws = trace.shape 
                        sgld_step = list(range(num_draws))
                        
                        # Plot individual chains
                        for i in range(num_chains):
                            draws = trace[i]
                            plt.plot(sgld_step, draws, linewidth=1, label=f"chain {i}")

                        # Plot mean with black dashed line
                        mean = np.mean(trace, axis=0)
                        plt.plot(
                            sgld_step,
                            mean,
                            color="black",
                            linestyle="--",
                            linewidth=2,
                            label="mean",
                            zorder=3,
                        )

                        # Add std shading
                        std = np.std(trace, axis=0)
                        plt.fill_between(
                            sgld_step, mean - std, mean + std, color="gray", alpha=0.3, zorder=2
                        )

                        # Formatting
                        llc_mean = stats.get('llc/mean', np.nan)
                        mala_acc = stats.get('mala/acceptance_rate', np.nan)
                        title = f"LLC Trace, avg LLC = {llc_mean:.4f}"
                        if not np.isnan(mala_acc):
                            title += f", MALA acceptance = {mala_acc:.3f}"
                        plt.title(title)
                        plt.xlabel("Step")
                        plt.ylabel("LLC")
                        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        # Save and show
                        if save_path:
                            plt.savefig(save_path, dpi=300, bbox_inches='tight')
                            print(f"Trace plot saved to {save_path}")
                        if show:
                            plt.show()
                        plt.close()
                    
                    # Call the custom plotting function
                    plot_sampling_evolution(llc_stats, save_path='llc_trace_test_run.png', show=False)
                    
                except Exception as plot_error:
                    print(f"Trace plotting failed: {plot_error}")
                    import traceback
                    traceback.print_exc()
            
            # Print recommended config values
            print(f"\n=== RECOMMENDED CONFIG VALUES ===")
            print(f"Update your config with:")
            print(f"  llc_epsilon = {epsilon:.2e}")
            print(f"  llc_gamma = {gamma:.3f}")
            print(f"  # Note: nbeta = {nbeta:.3f} (this goes in optimizer_kwargs)")
            
            return llc_stats
            
        except Exception as e:
            print(f"Test run failed: {e}")
            print("Falling back to manual calibration...")
            return None
    
    
    def _estimate_llc_for_calibration(self, model, evaluate, device, loader, 
                                    epsilon, beta):
        """Helper for calibration sweep"""
        try:
            # Use default_nbeta for proper beta calculation
            nbeta = default_nbeta(loader)
            stats = estimate_learning_coeff_with_summary(
                model=model,
                loader=loader,
                evaluate=evaluate,
                sampling_method=SGLD,
                optimizer_kwargs=dict(lr=epsilon, localization=1.0, nbeta=nbeta),  # Use proper nbeta
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




    # def _manual_calibration(self, model, dataloader, epsilon_range=None, gamma_range=None):
    #     """Manual grid search when EpsilonBetaAnalyzer fails"""
    #     print("Performing manual hyperparameter grid search...")
        
    #     # Use provided epsilon/gamma range or default
    #     if epsilon_range is None:
    #         epsilon_range = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
    #     if gamma_range is None:
    #         gamma_range = [0.1, 0.5, 1.0]
        
    #     # Get proper nbeta value using devinterp utility
    #     nbeta_standard = default_nbeta(dataloader)
    #     print(f"Default nbeta for dataloader: {nbeta_standard:.3f}")
    #     nbeta_range = [nbeta_standard]
        
    #     print(f"Testing {len(epsilon_range)} epsilon values, {len(nbeta_range)} nbeta values, {len(gamma_range)} gamma values")
        
    #     results = []
    #     successful_runs = 0
    #     total_runs = len(epsilon_range) * len(nbeta_range) * len(gamma_range)
        
    #     for epsilon in epsilon_range:
    #         for nbeta in nbeta_range:
    #             for gamma in gamma_range:
    #                 print(f"Testing epsilon={epsilon:.2e}, nbeta={nbeta:.3f}, gamma={gamma:.1f}")
                    
    #                 try:
    #                     stats = estimate_learning_coeff_with_summary(
    #                         model=model,
    #                         loader=dataloader,
    #                         evaluate=evaluate_ce,
    #                         sampling_method=SGLD,
    #                         optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta),
    #                         num_chains=2,  # Fewer chains for speed
    #                         num_draws=50,  # Fewer draws for stability
    #                         num_burnin_steps=45,  # 90% burn-in
    #                         device=self.device,
    #                         online=True,
    #                         verbose=False,
    #                     )
                        
    #                     llc_mean = stats.get('llc/mean', np.nan)
    #                     llc_std = stats.get('llc/std', np.nan)
                        
    #                     if not np.isnan(llc_mean):
    #                         results.append({
    #                             'epsilon': epsilon,
    #                             'nbeta': nbeta,
    #                             'gamma': gamma,
    #                             'llc_mean': llc_mean,
    #                             'llc_std': llc_std
    #                         })
    #                         successful_runs += 1
    #                         print(f"  ✓ LLC = {llc_mean:.4f} ± {llc_std:.4f}")
    #                     else:
    #                         print(f"  ✗ NaN result")
                            
    #                 except Exception as e:
    #                     print(f"  ✗ Error: {e}")
        
    #     # Print summary
    #     print(f"\n=== MANUAL CALIBRATION RESULTS ===")
    #     print(f"Successful runs: {successful_runs}/{total_runs}")
        
    #     if results:
    #         # Sort by LLC mean (lower is better for stability)
    #         results.sort(key=lambda x: x['llc_mean'])
            
    #         print(f"\nBest parameters (lowest LLC):")
    #         for i, result in enumerate(results[:5]):
    #             print(f"  {i+1}. ε={result['epsilon']:.2e}, nβ={result['nbeta']:.3f}, γ={result['gamma']:.1f}")
    #             print(f"     LLC = {result['llc_mean']:.4f} ± {result['llc_std']:.4f}")
            
    #         # Save results to file
    #         import json
    #         with open('llc_calibration_results.json', 'w') as f:
    #             json.dump(results, f, indent=2)
    #         print(f"\nResults saved to: llc_calibration_results.json")
            
    #         # Recommend best parameters
    #         best = results[0]
    #         print(f"\n=== RECOMMENDED PARAMETERS ===")
    #         print(f"Update your config with:")
    #         print(f"  llc_epsilon = {best['epsilon']:.2e}")
    #         print(f"  llc_gamma = {best['gamma']:.1f}")
    #         print(f"  # Note: nbeta = {best['nbeta']:.3f} (this goes in optimizer_kwargs)")
            
    #         # Test the best parameters with a single run
    #         print(f"\nTesting best parameters with single run...")
    #         test_stats = self._test_best_parameters(
    #             model, dataloader, 
    #             {'epsilon': best['epsilon'], 'gamma': best['gamma'], 'nbeta': best['nbeta']}
    #         )
            
    #     else:
    #         print("No successful runs found. Consider:")
    #         # print("  1. Using even smaller epsilon values")
    #         # print("  2. Checking model stability")
    #         # print("  3. Using different gamma values")
    #         # print("  4. Reducing the number of draws")
    #         print("  1. Using even smaller epsilon values (e.g., 1e-10)")
    #         print("  2. Using even smaller gamma values (e.g., 0.01)")
    #         print("  3. Reducing the number of draws to 20-30")
    #         print("  4. Trying an earlier checkpoint (less sharp minimum)")
    #         print("  5. Checking model stability with the diagnostic script")
        
    #     # Print current config values for reference
    #     print(f"\nCurrent config values:")
    #     print(f"  llc_epsilon: {self.epsilon:.2e}")
    #     print(f"  llc_gamma: {self.gamma:.2f}")
        
    #     return results
    