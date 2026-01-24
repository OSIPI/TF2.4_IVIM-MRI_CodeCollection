from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import importlib.util

class OSIPI_Statsmodels_WLS(OsipiBase):
    """
    Wrapper for Weighted Least Squares (WLS) fitting using Statsmodels.
    Based on Issue #110.
    """
    
    # IVIM data is usually 4D (x, y, z, b-values) or 2D (voxels, b-values)
    ACCEPTED_DIMENSIONS = (2, 4)
    
    def __init__(self, thresholds=None, bounds=None, initial_guess=None):
        """
        Initialize the wrapper and check for dependencies.
        """
        super(OSIPI_Statsmodels_WLS, self).__init__(
            thresholds=thresholds,
            bounds=bounds,
            initial_guess=initial_guess
        )
        
        # Check if statsmodels is installed
        if importlib.util.find_spec("statsmodels") is None:
            print("Warning: 'statsmodels' package not found. Please run: pip install statsmodels")

    def osipi_fit(self, data, bvalues):
        """
        Implementation of Segmented WLS fitting using statsmodels.RLM.
        """
        import statsmodels.api as sm
        
        # 1. Reshape Data: Flatten to (N_voxels, N_bvalues)
        original_shape = data.shape
        if data.ndim > 2:
            n_voxels = np.prod(data.shape[:-1])
            data_flat = data.reshape(n_voxels, data.shape[-1])
        elif data.ndim == 1:
            data_flat = data[np.newaxis, :]
            n_voxels = 1
        else:
            data_flat = data
            n_voxels = data.shape[0]

        # Prepare outputs
        f_map = np.zeros(n_voxels)
        D_star_map = np.zeros(n_voxels)
        D_map = np.zeros(n_voxels)
        S0_map = np.zeros(n_voxels)

        # 2. Define Thresholds for Segmented Fitting
        # b-values > 200 used for D (pure diffusion)
        # b-values < 200 used for D* (perfusion)
        b_threshold = 200.0
        high_b_mask = bvalues >= b_threshold
        low_b_mask = (bvalues < b_threshold) & (bvalues > 0) # Exclude 0 to avoid div/0
        
        # Find index of b=0 for normalization
        if 0 in bvalues:
            b0_idx = np.where(bvalues == 0)[0][0]
        else:
            b0_idx = None # Handle cases with no b=0 later

        # 3. Iterate over voxels
        for i in range(n_voxels):
            signal = data_flat[i, :]
            
            # A. Normalize Signal (S0)
            if b0_idx is not None:
                S0_val = signal[b0_idx]
            else:
                S0_val = np.max(signal) # Fallback
            
            if S0_val <= 0: 
                continue # Skip noise/background
            
            S0_map[i] = S0_val

            # --- STEP 1: Fit D and f using High b-values ---
            # Model: ln(S) approx ln(S0 * (1-f)) - b * D
            try:
                y_high = np.log(signal[high_b_mask])
                x_high = bvalues[high_b_mask]
                
                # Design Matrix: Column of 1s (intercept) and Column of -b (slope)
                X_high = sm.add_constant(-x_high)
                
                # Robust Linear Fit (RLM)
                model_D = sm.RLM(y_high, X_high, M=sm.robust.norms.HuberT())
                results_D = model_D.fit()
                
                intercept = results_D.params[0] # ln(S0 * (1-f))
                D_val = results_D.params[1]     # Slope is D
                
                # Calculate f from intercept
                # exp(intercept) = S0 * (1-f)  ->  f = 1 - (exp(intercept)/S0)
                f_val = 1.0 - (np.exp(intercept) / S0_val)
                
                # Constraints
                if D_val < 0: D_val = 0
                if f_val < 0: f_val = 0
                if f_val > 1: f_val = 1
                
                D_map[i] = D_val
                f_map[i] = f_val

                # --- STEP 2: Fit D* using Low b-values (Residuals) ---
                # We subtract the diffusion part we just found:
                # S_perfusion = S_measured - (S0 * (1-f) * exp(-b*D))
                # Model: ln(S_perfusion) approx ln(S0 * f) - b * D*
                
                diffusion_part = S0_val * (1 - f_val) * np.exp(-bvalues[low_b_mask] * D_val)
                perfusion_signal = signal[low_b_mask] - diffusion_part
                
                # Filter out negative residuals (log would fail)
                valid_perf = perfusion_signal > 0
                
                if np.sum(valid_perf) > 2:
                    y_low = np.log(perfusion_signal[valid_perf])
                    x_low = bvalues[low_b_mask][valid_perf]
                    
                    X_low = sm.add_constant(-x_low)
                    
                    model_Dstar = sm.RLM(y_low, X_low, M=sm.robust.norms.HuberT())
                    results_Dstar = model_Dstar.fit()
                    
                    D_star_val = results_Dstar.params[1]
                    if D_star_val < 0: D_star_val = 0
                    D_star_map[i] = D_star_val
                else:
                    D_star_map[i] = 0.0 # Not enough data for D* fit

            except Exception:
                # If fitting fails (e.g., log of negative noise), return zeros
                continue

        return f_map, D_star_map, D_map, S0_map