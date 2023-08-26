import numpy as np
import numpy.testing as npt
import pytest
import json
import pathlib

from utils.data_simulation.GenerateData import GenerateData
from src.original.ETP_SRI.LinearFitting import LinearFit


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
@pytest.mark.parametrize("D, bvals", test_linear_data)
def test_linear_fit(D, bvals):
    gd = GenerateData()
    gd_signal = gd.exponential_signal(D, bvals)
    print(gd_signal)
    fit = LinearFit()
    D_fit = fit.linear_fit(bvals, np.log(gd_signal))
    npt.assert_allclose([1, D], D_fit)

test_ivim_data = [
    pytest.param(0, 0.01, 0.05, np.linspace(0, 1000, 11), id='0'),
    pytest.param(0.1, 0.01, 0.05, np.linspace(0, 1000, 11), id='0.1'),
    pytest.param(0.2, 0.01, 0.05, np.linspace(0, 1000, 11), id='0.2'),
    pytest.param(0.1, 0.05, 0.1, np.linspace(0, 1000, 11), id='0.3'),
    pytest.param(0.4, 0.001, 0.05, np.linspace(0, 1000, 11), id='0.4'),
    pytest.param(0.5, 0.001, 0.05, np.linspace(0, 1000, 11), id='0.5'),
]
@pytest.mark.parametrize("f, D, Dp, bvals", test_ivim_data)
def test_ivim_fit(f, D, Dp, bvals):
    gd = GenerateData()
    gd_signal = gd.ivim_signal(D, Dp, f, 1, bvals)
    fit = LinearFit()
    [f_fit, D_fit, Dp_fit] = fit.ivim_fit(bvals, gd_signal)
    npt.assert_allclose([f, D], [f_fit, D_fit], atol=1e-5)
    if not np.allclose(f, 0):
        npt.assert_allclose(Dp, Dp_fit, rtol=1e-2, atol=1e-3)

def test_ivim_fit_saved(request):
    file = pathlib.Path(request.node.fspath)
    # print('current test file:', file)
    generic = file.with_name('generic.json')
    # print('current config file:', generic)
    fit = LinearFit()
    with generic.open() as f:
        data = json.load(f)
        bvals = data.pop('config')
        bvals = bvals['bvalues']
        for name, data in data.items():
            print(f'{name} {data}')
            signal = np.asarray(data['data'])
            signal /= signal[0]
            if data['f'] == 1.0:
                linear_fit = fit.linear_fit(bvals, np.log(signal))
                npt.assert_allclose([data['f'], data['Dp']], linear_fit, atol=1e-2)
            else:
                [f_fit, D_fit, Dp_fit] = fit.ivim_fit(bvals, signal)
                npt.assert_allclose([data['f'], data['D']], [f_fit, D_fit], atol=1e-2)
                npt.assert_allclose(data['Dp'], Dp_fit, atol=1e-1)  # go easy on the perfusion as it's a linear fake
