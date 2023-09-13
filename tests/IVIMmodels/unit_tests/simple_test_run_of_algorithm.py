import numpy as np
import os
from pathlib import Path
from src.standardized.ETP_SRI_LinearFitting import ETP_SRI_LinearFitting
from src.standardized.IAR_LU_biexp import IAR_LU_biexp

## Simple test code... 
# Used to just do a test run of an algorithm during development
def test_run(model, **kwargs):
    bvalues = np.array([0, 200, 500, 800])

    def ivim_model(b, S0=1, f=0.1, Dstar=0.03, D=0.001):
        return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

    signals = ivim_model(bvalues)

    #model = ETP_SRI_LinearFitting(thresholds=[200])
    if kwargs:
        results = model.osipi_fit(signals, bvalues, **kwargs)
    else:
        results = model.osipi_fit(signals, bvalues)
    print(results)
    #test = model.osipi_simple_bias_and_RMSE_test(SNR=20, bvalues=bvalues, f=0.1, Dstar=0.03, D=0.001, noise_realizations=10)
    
model1 = ETP_SRI_LinearFitting(thresholds=[200])
model2 = IAR_LU_biexp()

test_run(model1, linear_fit_option=True)
test_run(model2)

###########################################

path = Path(__file__).resolve().parents[3] # Move up to the root folder
path_standardized_algortihms = path / "src" / "standardized" # Move to the folder containing the algorithms
algorithms = os.listdir(path_standardized_algortihms) # Get the contents of the folder

# Remove some crap
algorithms.remove("__init__.py")
algorithms.remove("__pycache__")
algorithms.remove("template.py")

# Remove the .py extensions from the algorithms names
algorithms = [algorithm.split(".")[0] for algorithm in algorithms]