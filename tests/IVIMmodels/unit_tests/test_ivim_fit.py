import numpy as np
import numpy.testing as npt
import pytest
import json
import pathlib
import os

from src.wrappers.OsipiBase import OsipiBase
from utilities.data_simulation.GenerateData import GenerateData

#run using python -m pytest from the root folder

test_linear_data = [
    pytest.param(0, np.linspace(0, 1000, 11), id='0'),
    pytest.param(0.01, np.linspace(0, 1000, 11), id='0.1'),
    pytest.param(0.02, np.linspace(0, 1000, 11), id='0.2'),
    pytest.param(0.03, np.linspace(0, 1000, 11), id='0.3'),
    pytest.param(0.04, np.linspace(0, 1000, 11), id='0.4'),
    pytest.param(0.05, np.linspace(0, 1000, 11), id='0.5'),
    pytest.param(0.08, np.linspace(0, 1000, 11), id='0.8'),
    pytest.param(0.1, np.linspace(0, 1000, 11), id='1'),
]

#@pytest.mark.parametrize("D, bvals", test_linear_data)
#def test_linear_fit(D, bvals):
    #gd = GenerateData()
    #gd_signal = gd.exponential_signal(D, bvals)
    #print(gd_signal)
    #fit = LinearFit()
    #D_fit = fit.linear_fit(bvals, np.log(gd_signal))
    #npt.assert_allclose([1, D], D_fit)

test_ivim_data = [
    pytest.param(0, 0.01, 0.05, np.linspace(0, 1000, 11), id='0'),
    pytest.param(0.1, 0.01, 0.05, np.linspace(0, 1000, 11), id='0.1'),
    pytest.param(0.2, 0.01, 0.05, np.linspace(0, 1000, 11), id='0.2'),
    pytest.param(0.1, 0.05, 0.1, np.linspace(0, 1000, 11), id='0.3'),
    pytest.param(0.4, 0.001, 0.05, np.linspace(0, 1000, 11), id='0.4'),
    pytest.param(0.5, 0.001, 0.05, np.linspace(0, 1000, 11), id='0.5'),
]

#@pytest.mark.parametrize("f, D, Dp, bvals", test_ivim_data)
#def test_ivim_fit(f, D, Dp, bvals):
    ## We should make a wrapper that runs this for a range of different settings, such as b thresholds, bounds, etc.
    ## An additional inputs to these functions could perhaps be a "settings" class with attributes that are the settings to the
    ## algorithms. I.e. bvalues, thresholds, bounds, initial guesses.
    ## That way, we can write something that defines a range of settings, and then just run them through here.
    
    #gd = GenerateData()
    #gd_signal = gd.ivim_signal(D, Dp, f, 1, bvals)
    
    ##fit = LinearFit()   # This is the old code by ETP
    #fit = ETP_SRI_LinearFitting() # This is the standardized format by IAR, which every algorithm will be implemented with
    
    #[f_fit, Dp_fit, D_fit] = fit.ivim_fit(gd_signal, bvals) # Note that I have transposed Dp and D. We should decide on a standard order for these. I usually go with f, Dp, and D ordered after size.
    #npt.assert_allclose([f, D], [f_fit, D_fit], atol=1e-5)
    #if not np.allclose(f, 0):
        #npt.assert_allclose(Dp, Dp_fit, rtol=1e-2, atol=1e-3)

def data_ivim_fit_saved():
    # Find the algorithm file in the folder structure
    path = pathlib.Path(__file__).resolve().parents[3] # Move up to the root folder
    path_standardized_algortihms = path / "src" / "standardized" # Move to the folder containing the algorithms
    algorithms = os.listdir(path_standardized_algortihms) # Get the contents of the folder

    # Remove the .py extensions from the algorithm names
    algorithms = [algorithm for algorithm in algorithms if algorithm[0:2].isupper()]
    algorithms = [algorithm.split(".")[0] for algorithm in algorithms]
    
     
    file = pathlib.Path(__file__)
    generic = file.with_name('generic.json')
    with generic.open() as f:
        all_data = json.load(f)
        bvals = all_data.pop('config')
        bvals = bvals['bvalues']
        for name, data in all_data.items():
            for algorithm in algorithms:
                yield name, bvals, data, algorithm
            
            
@pytest.mark.parametrize("name, bvals, data, algorithm", data_ivim_fit_saved())
def test_ivim_fit_saved(name, bvals, data, algorithm):
    fit = OsipiBase(algorithm=algorithm, thresholds=[500])
    signal = np.asarray(data['data'])
    has_negatives = np.any(signal<0)
    signal = np.abs(signal)
    ratio = 1 / signal[0]
    signal /= signal[0]
    tolerance = 1e-2 + 25 * data['noise'] * ratio  # totally empirical
    #if has_negatives:  # this fitting doesn't do well with negatives
    #    tolerance += 1
    #if data['f'] == 1.0:
        #linear_fit = fit.ivim_fit(np.log(signal), bvals, linear_fit_option=True)
        #npt.assert_allclose([data['f'], data['Dp']], linear_fit, atol=tolerance)
    #else:
        #[f_fit, Dp_fit, D_fit] = fit.ivim_fit(signal, bvals)
        #npt.assert_allclose([data['f'], data['D']], [f_fit, D_fit], atol=tolerance)
        #npt.assert_allclose(data['Dp'], Dp_fit, atol=1e-1)  # go easy on the perfusion as it's a linear fake

    [f_fit, Dp_fit, D_fit] = fit.ivim_fit(signal, bvals)
    npt.assert_allclose([data['f'], data['D']], [f_fit, D_fit], atol=tolerance)
    npt.assert_allclose(data['Dp'], Dp_fit, atol=1e-1)  # go easy on the perfusion as it's a linear fake
