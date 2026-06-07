import numpy as np
import pytest

# Import from run_cp3_validation (which we will create next)
from run_cp3_validation import compute_jacobian, compute_crlb, fit_biexp_nlls

def test_jacobian_against_finite_differences():
    bvals = np.array([0, 50, 100, 200, 400, 600, 800, 1000], float)
    # Test at a few points in the prior range
    test_points = [
        [1.1e-3, 30e-3, 0.15],
        [0.8e-3, 10e-2, 0.3],
        [2.5e-3, 5e-3, 0.05]
    ]
    eps = 1e-9
    for theta in test_points:
        J_anal = compute_jacobian(theta, bvals)
        
        # Central finite difference approximation
        from ivim_simulator import ivim_signal
        J_cfd = np.zeros((len(bvals), 3))
        for i in range(3):
            theta_plus = np.array(theta, dtype=float)
            theta_plus[i] += eps
            theta_minus = np.array(theta, dtype=float)
            theta_minus[i] -= eps
            pred_plus = ivim_signal(bvals, *theta_plus)
            pred_minus = ivim_signal(bvals, *theta_minus)
            J_cfd[:, i] = (pred_plus - pred_minus) / (2 * eps)
            
        assert np.allclose(J_anal, J_cfd, atol=1e-5), f"Jacobian mismatch at {theta}"

def test_crlb_finite_positive():
    bvals = np.array([0, 50, 100, 200, 400, 600, 800, 1000], float)
    theta = [1.1e-3, 30e-3, 0.15]
    snr = 20.0
    sqrt_crlb = compute_crlb(theta, bvals, snr)
    
    assert sqrt_crlb.shape == (3,)
    assert np.all(np.isfinite(sqrt_crlb))
    assert np.all(sqrt_crlb > 0.0)

def test_nlls_fit_convergence():
    bvals = np.array([0, 50, 100, 200, 400, 600, 800, 1000], float)
    theta_true = np.array([1.1e-3, 30e-3, 0.15])
    
    from ivim_simulator import ivim_signal
    clean_signal = ivim_signal(bvals, *theta_true)
    
    # Fit the clean signal (should converge close to truth)
    fit_params = fit_biexp_nlls(bvals, clean_signal)
    
    assert fit_params.shape == (3,)
    assert np.allclose(fit_params, theta_true, rtol=1e-3, atol=1e-5)
