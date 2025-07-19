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
        """Calibrate hyperparameters using EpsilonBetaAnalyzer"""
        print("Starting hyperparameter calibration...")
        
        # # Calculate model size for parameter selection
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Model has {total_params:,} parameters")
        
        # # Determine epsilon range based on model size
        # if total_params <= 10_000:
        #     epsilon_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # Small networks
        # elif total_params <= 1_000_000:
        #     epsilon_range = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]  # Medium networks (ResNet18 is ~11M)
        # else:
        #     epsilon_range = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]  # Large networks
        
        # print(f"Using epsilon range: {[f'{e:.1e}' for e in epsilon_range]})

        # Use very conservative ranges for sharp minima
        epsilon_range = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
        gamma_range = [0.1, 0.5, 1.0]
        print(f"Using epsilon range: {[f'{e:.1e}' for e in epsilon_range]}")
        print(f"Using gamma range: {gamma_range}")
        
        # Try the EpsilonBetaAnalyzer approach first with timeout
        try:
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("EpsilonBetaAnalyzer timed out")
            
            # Set timeout for the analyzer (5 minutes)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)
            
            # Initialize the analyzer
            analyzer = EpsilonBetaAnalyzer()
            
            # Configure the sweep with conservative ranges
            analyzer.configure_sweep(
                llc_estimator=self._estimate_llc_for_calibration,
                llc_estimator_kwargs=dict(
                    model=model,
                    evaluate=evaluate_ce,
                    device=self.device,
                    loader=dataloader,
                ),
                min_epsilon=epsilon_range[0],
                max_epsilon=epsilon_range[-1],
                epsilon_samples=len(epsilon_range), #used to be 5
                beta_samples=3, #used to be 5
                dataloader=dataloader,
            )
            
            # Run the sweep
            analyzer.sweep()
            
            # Cancel timeout for plotting
            signal.alarm(0)
            
            # Try to extract best parameters from analyzer
            best_params = None
            try:
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
                        # Find best parameters: lowest LLC with reasonable stability
                        # Sort by LLC value (lower is better) but penalize high variance
                        valid_results['stability_score'] = valid_results['llc/final'] * (1 + valid_results['llc/std_over_mean'])
                        best_idx = valid_results['stability_score'].idxmin()
                        best_row = valid_results.loc[best_idx]
                        
                        best_params = {
                            'epsilon': best_row['epsilon'],
                            'beta': best_row['beta'],
                            'llc_mean': best_row['llc/final'],
                            'llc_std_over_mean': best_row['llc/std_over_mean']
                        }
                        
                        print(f"Best parameters found:")
                        print(f"  ε = {best_params['epsilon']:.2e}")
                        print(f"  β = {best_params['beta']:.3f}")
                        print(f"  LLC = {best_params['llc_mean']:.4f}")
                        print(f"  Stability (std/mean) = {best_params['llc_std_over_mean']:.4f}")
                        
                        # Also print top 5 results for comparison
                        print(f"\nTop 5 parameter combinations:")
                        top_5 = valid_results.nsmallest(5, 'stability_score')
                        for i, (_, row) in enumerate(top_5.iterrows()):
                            print(f"  {i+1}. ε={row['epsilon']:.2e}, β={row['beta']:.3f}, LLC={row['llc/final']:.4f}, stability={row['llc/std_over_mean']:.4f}")
                    else:
                        print("No valid results found in analyzer")
                else:
                    print("No sweep results available")
            except Exception as e:
                print(f"Warning: Error extracting best parameters: {e}")
                import traceback
                traceback.print_exc()
            
            # Save the plots with timeout
            signal.alarm(60)  # 1 minute timeout for plotting
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))
                analyzer.plot()
                plt.savefig('llc_calibration_results.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Calibration plots saved to llc_calibration_results.png")
            except Exception as plot_error:
                print(f"Plotting failed: {plot_error}")
                print("Continuing with results...")
            finally:
                signal.alarm(0)
            
            # If we got best parameters, do a single test run
            if best_params is not None:
                print(f"\nBest parameters from analyzer: {best_params}")
                self._test_best_parameters(model, dataloader, best_params)
            else:
                print("No best parameters extracted, falling back to manual search...")
                return self._manual_calibration(model, dataloader, epsilon_range, gamma_range)
            
            return analyzer
            
        except TimeoutError:
            print("EpsilonBetaAnalyzer timed out, falling back to manual grid search...")
            signal.alarm(0)  # Cancel any remaining timeout
            return self._manual_calibration(model, dataloader, epsilon_range, gamma_range)
        except Exception as e:
            print(f"EpsilonBetaAnalyzer failed: {e}")
            print("Falling back to manual grid search...")
            signal.alarm(0)  # Cancel any remaining timeout
            return self._manual_calibration(model, dataloader, epsilon_range, gamma_range)
    
    def _test_best_parameters(self, model, dataloader, best_params):
        """Test the best parameters with a single run and plot trace"""
        print("\n=== TESTING BEST PARAMETERS ===")
        
        # Extract parameters (handle different formats)
        if isinstance(best_params, dict):
            epsilon = best_params.get('epsilon', best_params.get('lr', 1e-8))
            # For analyzer results, 'beta' is the nbeta value
            nbeta = best_params.get('beta', best_params.get('nbeta', default_nbeta(dataloader)))
            gamma = best_params.get('gamma', best_params.get('localization', 1.0))
        else:
            # Assume it's a tuple or list
            epsilon = best_params[0] if len(best_params) > 0 else 1e-8
            nbeta = best_params[1] if len(best_params) > 1 else default_nbeta(dataloader)
            gamma = best_params[2] if len(best_params) > 2 else 1.0
        
        print(f"Testing with: ε={epsilon:.2e}, nβ={nbeta:.3f}, γ={gamma:.2f}")
        
        try:
            # Make a single test run with chosen hyperparameters
            print("Making single test run with best hyperparameters...")
            llc_stats = estimate_learning_coeff_with_summary(
                model=model,
                loader=dataloader,
                evaluate=evaluate_ce,
                sampling_method=SGLD,
                optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta),
                num_chains=2,
                num_draws=50,
                num_burnin_steps=45,
                device=self.device,
                online=True,
                verbose=True,
            )
            
            # Print results
            llc_mean = llc_stats.get('llc/mean', np.nan)
            llc_std = llc_stats.get('llc/std', np.nan)
            print(f"Test run result: LLC = {llc_mean:.4f} ± {llc_std:.4f}")
            
            # Plot the trace if available
            if 'llc/trace' in llc_stats:
                try:
                    from devinterp.utils import plot_trace
                    import matplotlib.pyplot as plt
                    
                    print("Plotting LLC trace...")
                    plot_trace(
                        llc_stats['llc/trace'],
                        "LLC Loss",
                        x_axis="Step",
                        title=f"LLC Trace, avg LLC = {llc_mean:.4f} ± {llc_std:.4f}",
                        plot_mean=False,
                        plot_std=False,
                        fig_size=(12, 9),
                        true_lc=None,
                    )
                    plt.savefig('llc_trace_test_run.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print("Trace plot saved to llc_trace_test_run.png")
                    
                except Exception as plot_error:
                    print(f"Trace plotting failed: {plot_error}")
            
            # Print recommended config values
            print(f"\n=== RECOMMENDED CONFIG VALUES ===")
            print(f"Update your config with:")
            print(f"  llc_epsilon = {epsilon:.2e}")
            print(f"  llc_gamma = {gamma:.2f}")
            print(f"  # Note: nbeta = {nbeta:.3f} (this goes in optimizer_kwargs)")
            
            return llc_stats
            
        except Exception as e:
            print(f"Test run failed: {e}")
            print("Falling back to manual calibration...")
            return None
    
    def _manual_calibration(self, model, dataloader, epsilon_range=None, gamma_range=None):
        """Manual grid search when EpsilonBetaAnalyzer fails"""
        print("Performing manual hyperparameter grid search...")
        
        # Use provided epsilon/gamma range or default
        if epsilon_range is None:
            epsilon_range = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7]
        if gamma_range is None:
            gamma_range = [0.1, 0.5, 1.0]
        
        # Get proper nbeta value using devinterp utility
        nbeta_standard = default_nbeta(dataloader)
        print(f"Default nbeta for dataloader: {nbeta_standard:.3f}")
        nbeta_range = [nbeta_standard]
        
        print(f"Testing {len(epsilon_range)} epsilon values, {len(nbeta_range)} nbeta values, {len(gamma_range)} gamma values")
        
        results = []
        successful_runs = 0
        total_runs = len(epsilon_range) * len(nbeta_range) * len(gamma_range)
        
        for epsilon in epsilon_range:
            for nbeta in nbeta_range:
                for gamma in gamma_range:
                    print(f"Testing epsilon={epsilon:.2e}, nbeta={nbeta:.3f}, gamma={gamma:.1f}")
                    
                    try:
                        stats = estimate_learning_coeff_with_summary(
                            model=model,
                            loader=dataloader,
                            evaluate=evaluate_ce,
                            sampling_method=SGLD,
                            optimizer_kwargs=dict(lr=epsilon, localization=gamma, nbeta=nbeta),
                            num_chains=2,  # Fewer chains for speed
                            num_draws=50,  # Fewer draws for stability
                            num_burnin_steps=45,  # 90% burn-in
                            device=self.device,
                            online=True,
                            verbose=False,
                        )
                        
                        llc_mean = stats.get('llc/mean', np.nan)
                        llc_std = stats.get('llc/std', np.nan)
                        
                        if not np.isnan(llc_mean):
                            results.append({
                                'epsilon': epsilon,
                                'nbeta': nbeta,
                                'gamma': gamma,
                                'llc_mean': llc_mean,
                                'llc_std': llc_std
                            })
                            successful_runs += 1
                            print(f"  ✓ LLC = {llc_mean:.4f} ± {llc_std:.4f}")
                        else:
                            print(f"  ✗ NaN result")
                            
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
        
        # Print summary
        print(f"\n=== MANUAL CALIBRATION RESULTS ===")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        
        if results:
            # Sort by LLC mean (lower is better for stability)
            results.sort(key=lambda x: x['llc_mean'])
            
            print(f"\nBest parameters (lowest LLC):")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. ε={result['epsilon']:.2e}, nβ={result['nbeta']:.3f}, γ={result['gamma']:.1f}")
                print(f"     LLC = {result['llc_mean']:.4f} ± {result['llc_std']:.4f}")
            
            # Save results to file
            import json
            with open('llc_calibration_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: llc_calibration_results.json")
            
            # Recommend best parameters
            best = results[0]
            print(f"\n=== RECOMMENDED PARAMETERS ===")
            print(f"Update your config with:")
            print(f"  llc_epsilon = {best['epsilon']:.2e}")
            print(f"  llc_gamma = {best['gamma']:.1f}")
            print(f"  # Note: nbeta = {best['nbeta']:.3f} (this goes in optimizer_kwargs)")
            
            # Test the best parameters with a single run
            print(f"\nTesting best parameters with single run...")
            test_stats = self._test_best_parameters(
                model, dataloader, 
                {'epsilon': best['epsilon'], 'gamma': best['gamma'], 'nbeta': best['nbeta']}
            )
            
        else:
            print("No successful runs found. Consider:")
            # print("  1. Using even smaller epsilon values")
            # print("  2. Checking model stability")
            # print("  3. Using different gamma values")
            # print("  4. Reducing the number of draws")
            print("  1. Using even smaller epsilon values (e.g., 1e-10)")
            print("  2. Using even smaller gamma values (e.g., 0.01)")
            print("  3. Reducing the number of draws to 20-30")
            print("  4. Trying an earlier checkpoint (less sharp minimum)")
            print("  5. Checking model stability with the diagnostic script")
        
        # Print current config values for reference
        print(f"\nCurrent config values:")
        print(f"  llc_epsilon: {self.epsilon:.2e}")
        print(f"  llc_gamma: {self.gamma:.2f}")
        
        return results
    
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