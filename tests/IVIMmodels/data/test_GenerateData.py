import numpy as np
import numpy.testing as npt
import pytest
#import torch

from utilities.data_simulation.GenerateData import GenerateData

#run using python -m pytest from the root folder

test_monoexponential_data = [
    pytest.param(0, np.linspace(0, 1000, 11), id='0'),
    pytest.param(0.1, np.linspace(0, 1000, 11), id='0.1'),
    pytest.param(0.2, np.linspace(0, 1000, 11), id='0.2'),
    pytest.param(0.3, np.linspace(0, 1000, 11), id='0.3'),
    pytest.param(0.4, np.linspace(0, 1000, 11), id='0.4'),
    pytest.param(0.5, np.linspace(0, 1000, 11), id='0.5'),
    pytest.param(0.8, np.linspace(0, 1000, 11), id='0.8'),
    pytest.param(1, np.linspace(0, 1000, 11), id='1'),
]
@pytest.mark.parametrize("D, bvals", test_monoexponential_data)
def test_monoexponential(D, bvals):
    gd = GenerateData()
    gd_signal = gd.exponential_signal(D, bvals)
    testing_signal = np.exp(-D * np.asarray(bvals, dtype='float64'))
    npt.assert_allclose(gd_signal, testing_signal)
    assert(gd_signal[0] >= testing_signal[0])

test_ivim_data = [
    pytest.param(0.01, 0.0, 1, 1, np.linspace(0, 1000, 11), None, False),
    pytest.param(0.01, 0.1, 0.05, 1, np.linspace(0, 1000, 11), None, False),
    pytest.param(0.05, 0.2, 0.1, 1, np.linspace(0, 800, 11), None, False),
    pytest.param(0.04, 0.15, 0.25, 1.5, np.linspace(0, 1000, 2), 10, True),
    pytest.param(0.1, 0.5, 0.5, 0.5, np.linspace(0, 1500, 5), 100, True),
    pytest.param(0.01, 0.2, 0.1, 1, np.linspace(10, 1500, 11), 100, False),
    pytest.param(0.1, 0.15, 0.05, 1, np.linspace(10, 1000, 8), 5, False)
]
@pytest.mark.parametrize('D, Dp, f, S0, bvals, snr, rician_noise', test_ivim_data)
def test_ivim(D, Dp, f, S0, bvals, snr, rician_noise):
    gd = GenerateData()
    gd_signal = gd.ivim_signal(D, Dp, f, S0, bvals, snr, rician_noise)
    testing_signal = S0 * ((1 - f) * np.exp(-D * bvals) + f * np.exp(-Dp * bvals))
    atol = 0.0
    if snr is not None:
        atol = 4 / snr
    npt.assert_allclose(gd_signal, testing_signal, atol=atol)


test_linear_data = [
    pytest.param(0, np.linspace(0, 1000, 11), 0, id='0'),
    pytest.param(0.1, np.linspace(0, 1000, 11), 10, id='0.1'),
    pytest.param(0.2, np.linspace(0, 1000, 11), -10, id='0.2'),
    pytest.param(0.3, np.linspace(0, 1000, 11), 0, id='0.3'),
    pytest.param(0.4, np.linspace(0, 1000, 11), 0, id='0.4'),
    pytest.param(0.5, np.linspace(0, 1000, 11), 0, id='0.5'),
    pytest.param(0.8, np.linspace(0, 1000, 11), 0, id='0.8'),
    pytest.param(1, np.linspace(0, 1000, 11), 0, id='1'),
]
@pytest.mark.parametrize("D, bvals, offset", test_linear_data)
def test_linear(D, bvals, offset):
    gd = GenerateData()
    gd_signal = gd.linear_signal(D, bvals, offset)
    testing_signal = -D * np.asarray(bvals, dtype='float64')
    testing_signal += offset
    npt.assert_allclose(gd_signal, testing_signal)
    assert(gd_signal[0] >= testing_signal[0])
        
    gd_exponential = gd.exponential_signal(D, bvals)
    gd_log_exponential = np.log(gd_exponential) + offset
    real_mask = np.isfinite(gd_log_exponential)
    
    npt.assert_allclose(gd_log_exponential[real_mask], gd_signal[real_mask])
