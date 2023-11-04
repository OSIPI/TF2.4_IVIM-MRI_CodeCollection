import numpy as np
import os
from pathlib import Path
#from src.standardized.ETP_SRI_LinearFitting import ETP_SRI_LinearFitting
from src.standardized.IAR_LU_biexp import IAR_LU_biexp
#from src.standardized.IAR_LU_segmented_2step import IAR_LU_segmented_2step
from src.standardized.PvH_KB_NKI_IVIMfit import PvH_KB_NKI_IVIMfit
#from src.standardized.PV_MUMC_biexp import PV_MUMC_biexp

## Simple test code... 
# Used to just do a test run of an algorithm during development
def dev_test_run(model, **kwargs):
    bvalues = np.array([0, 50, 100, 150, 200, 500, 800])

    def ivim_model(b, S0=1, f=0.1, Dstar=0.01, D=0.001):
        return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

    signals = ivim_model(bvalues)

    #model = ETP_SRI_LinearFitting(thresholds=[200])
    if kwargs:
        results = model.osipi_fit(signals, bvalues, **kwargs)
    else:
        results = model.osipi_fit(signals, bvalues)
    print(results)
    #test = model.osipi_simple_bias_and_RMSE_test(SNR=20, bvalues=bvalues, f=0.1, Dstar=0.03, D=0.001, noise_realizations=10)
    
#model1 = ETP_SRI_LinearFitting(thresholds=[200])
#model2 = IAR_LU_biexp()
model2 = PvH_KB_NKI_IVIMfit()

#dev_test_run(model1, linear_fit_option=True)
dev_test_run(model2)
