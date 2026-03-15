import pytest
import numpy as np
from src.standardized.OSIPI_Statsmodels_WLS import OSIPI_Statsmodels_WLS

class TestStatsmodelsWLS:
    def test_wls_fitting_accuracy(self):
        # 1. Generate Synthetic Data (Ground Truth)
        # Standard b-values for an IVIM experiment
        bvalues = np.array([0, 10, 20, 50, 100, 200, 400, 800])
        
        # True Parameters (What we expect the algorithm to find)
        S0_true = 1000
        f_true = 0.2
        D_star_true = 0.05  # Fast diffusion (Perfusion)
        D_true = 0.001      # Slow diffusion
        
        # IVIM Signal Equation: S = S0 * ( f*exp(-b*D*) + (1-f)*exp(-b*D) )
        signal = S0_true * (
            f_true * np.exp(-bvalues * D_star_true) + 
            (1 - f_true) * np.exp(-bvalues * D_true)
        )
        
        # 2. Initialize the Wrapper
        model = OSIPI_Statsmodels_WLS()
        
        # 3. Fit the Data
        # Input shape must be (Voxels, b-values), so we add a dimension
        data_input = signal[np.newaxis, :]
        
        f_est, D_star_est, D_est, S0_est = model.osipi_fit(data_input, bvalues)
        
        # 4. Print & Assert (Verify the results are close to truth)
        print(f"\nGround Truth: f={f_true}, D*={D_star_true}, D={D_true}, S0={S0_true}")
        print(f"Estimated:    f={f_est[0]:.4f}, D*={D_star_est[0]:.4f}, D={D_est[0]:.4f}, S0={S0_est[0]:.4f}")
        
        # Check D (Diffusion) - Should be very accurate
        assert np.isclose(D_est[0], D_true, atol=0.0005), "D estimation failed"
        
        # Check S0 - Should be very accurate
        assert np.isclose(S0_est[0], S0_true, rtol=0.1), "S0 estimation failed"
        
        # Check f (Perfusion fraction) - Allow small error (segmented fitting approximation)
        assert np.isclose(f_est[0], f_true, atol=0.1), "f estimation failed"

        # Check D* (Pseudo-diffusion) - This is the hardest parameter, allow wider margin
        assert np.isclose(D_star_est[0], D_star_true, atol=0.04), "D* estimation failed"