import torch
import numpy as np
from torch.utils.data import DataLoader
from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce

class LLCEstimator:
    def __init__(self, config):
        self.epsilon = config.llc_epsilon  # SGLD step size
        self.gamma = config.llc_gamma      # localization parameter 
        self.num_chains = config.llc_num_chains
        self.num_draws = config.llc_num_draws
        self.num_burnin = config.llc_num_burnin
        self.device = config.device
        
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
        """One-time hyperparameter calibration"""
        analyzer = EpsilonBetaAnalyzer()
        analyzer.configure_sweep(
            llc_estimator=self._estimate_llc_for_calibration,
            llc_estimator_kwargs=dict(
                model=model,
                evaluate=evaluate_ce,
                device=self.device,
                loader=dataloader,
            ),
            min_epsilon=1e-4,
            max_epsilon=1e-1,
            epsilon_samples=5,
            beta_samples=5,
            dataloader=dataloader,
        )
        analyzer.sweep()
        analyzer.plot()  # Save calibration plot
        return analyzer
    
    def _estimate_llc_for_calibration(self, model, evaluate, device, loader, 
                                    epsilon, beta):
        """Helper for calibration sweep"""
        return estimate_learning_coeff_with_summary(
            model=model,
            loader=loader,
            evaluate=evaluate,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=epsilon, localization=5.0, nbeta=beta),
            num_chains=2,
            num_draws=200,  # Faster for calibration
            device=device,
            online=True,
            verbose=False,
        )