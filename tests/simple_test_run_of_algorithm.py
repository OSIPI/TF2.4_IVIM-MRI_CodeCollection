import numpy as np
from src.standardized.ETP_SRI_LinearFitting import ETP_SRI_LinearFitting
from src.standardized.IAR_LU_biexp import IAR_LU_biexp

## Simple test code... 
# Used to just do a test run of an algorithm during development
def test_run(model):
    bvalues = np.array([0, 200, 500, 800])

    def ivim_model(b, S0=1, f=0.1, Dstar=0.03, D=0.001):
        return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

    signals = ivim_model(bvalues)

    #model = ETP_SRI_LinearFitting(thresholds=[200])
    results = model.osipi_fit(signals, bvalues)
    print(results)
    #test = model.osipi_simple_bias_and_RMSE_test(SNR=20, bvalues=bvalues, f=0.1, Dstar=0.03, D=0.001, noise_realizations=10)
    
model1 = ETP_SRI_LinearFitting(thresholds=[200])
model2 = IAR_LU_biexp()

test_run(model1)
test_run(model2)