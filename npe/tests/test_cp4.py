import numpy as np
import torch
import pytest

from run_cp4_atlas import compute_metrics, classify_point

def test_compute_metrics():
    # Create mock posterior samples
    # Shape: (S, 3) where parameters are [D, Dstar, f]
    S = 1000
    # True values
    D_samples = torch.normal(1.1e-3, 0.1e-3, size=(S,))
    Dstar_samples = torch.normal(30e-3, 5e-3, size=(S,))
    f_samples = torch.normal(0.15, 0.02, size=(S,))
    samples = torch.stack([D_samples, Dstar_samples, f_samples], dim=1)
    
    prior_low = torch.tensor([0.2e-3, 3.0e-3, 0.0])
    prior_high = torch.tensor([3.0e-3, 0.15, 0.5])
    
    shrinkage, corr, collapse = compute_metrics(samples, prior_low, prior_high)
    
    assert shrinkage.shape == (3,)
    assert np.isscalar(corr)
    assert np.isscalar(collapse)
    
    # Shrinkage should be high since variance is small relative to prior
    assert np.all(shrinkage > 0.5)
    # Collapse indicator should be close to 0 since f is centered at 0.15
    assert collapse < 0.01

def test_classification():
    # Case 1: Recoverable (high shrinkage, low correlation, low collapse)
    class1 = classify_point(
        shrinkage_Dstar=0.7,
        correlation=0.1,
        collapse_indicator=0.01
    )
    assert class1 == "recoverable"
    
    # Case 2: Mono-exponential collapse (high collapse)
    class2 = classify_point(
        shrinkage_Dstar=0.1,
        correlation=0.1,
        collapse_indicator=0.6
    )
    assert class2 == "mono-exp collapse"
    
    # Case 3: f-Dstar trade-off (low shrinkage on Dstar)
    class3 = classify_point(
        shrinkage_Dstar=0.3,
        correlation=0.2,
        collapse_indicator=0.02
    )
    assert class3 == "f-D* trade-off"
    
    # Case 4: f-Dstar trade-off (high correlation)
    class4 = classify_point(
        shrinkage_Dstar=0.8,
        correlation=-0.6,
        collapse_indicator=0.02
    )
    assert class4 == "f-D* trade-off"
