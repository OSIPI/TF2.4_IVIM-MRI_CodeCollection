
import sys
# sys.path.append(r"C:\Users\ivan5\Box\OSIPI\TF2.4_IVIM-MRI_CodeCollection")
sys.path.append(f".")


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from src.standardized.ETP_SRI_LinearFitting import ETP_SRI_LinearFitting

# from src.standardized.IAR_LU_biexp import IAR_LU_biexp

# from src.standardized.IAR_LU_biexp import IAR_LU_biexp
# from src.standardized.IAR_LU_modified_mix import IAR_LU_modified_mix

#from src.standardized.IAR_LU_segmented_2step import IAR_LU_segmented_2step
#from src.standardized.PvH_KB_NKI_IVIMfit import PvH_KB_NKI_IVIMfit
#from src.standardized.PV_MUMC_biexp import PV_MUMC_biexp

from src.original.ETP_SRI.Sampling import MCMC
from src.standardized.ETP_SRI_MCMC import ETP_SRI_MCMC

from src.standardized.OGC_AmsterdamUMC_biexp import OGC_AmsterdamUMC_biexp


## Simple test code... 
# Used to just do a test run of an algorithm during development
def dev_test_run(model, **kwargs):
    bvalues = np.array([0, 20, 50, 75, 100, 150, 200, 300, 400, 500, 800, 1000, 1500])

    def ivim_model(b, f=0.1, Dstar=0.01, D=0.001, S0=1):
        # print(f'S0 {S0}')
        # print(f'Dstar {f*np.exp(-b*Dstar)}')
        # print(f'D {(1-f)*np.exp(-b*D)}')
        # print(f'sum {f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D)}')
        # print(f'S0 {(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))}')
        return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))


    # TODO: add S0 fitting!
    true_f = 0.4
    true_Dstar = 0.01
    true_D = 0.001
    truth = [true_f, true_D, true_Dstar]
    signals_noiseless = ivim_model(bvalues, true_f, true_Dstar, true_D)
    print(f'noiselss {signals_noiseless}')
    signals = signals_noiseless + np.abs(1e-1 * (np.random.randn(len(bvalues)) + 1j * np.random.randn(len(bvalues))) / np.sqrt(2))

    # signals = ivim_model(bvalues)
    # data = np.array([signals, signals, signals])
    # #print(data)
    # signals = data


    #model = ETP_SRI_LinearFitting(thresholds=[200])
    if kwargs:
        results = model.osipi_fit(signals, bvalues, **kwargs)
    else:
        results = model.osipi_fit(signals, bvalues)
    # print(results) # f, D*, D
    results_reordered = np.asarray([results['f'], results['Dp'], results['D']])
    print(truth)
    print(results_reordered)
    #test = model.osipi_simple_bias_and_RMSE_test(SNR=20, bvalues=bvalues, f=0.1, Dstar=0.03, D=0.001, noise_realizations=10)
    signal_results = ivim_model(bvalues, results['f'], results['Dp'], results['D'])

    

    # mcmc = MCMC(signals, bvalues, gaussian_noise=True, data_scale=1e-2) #, priors=((0.07, 1e-1), (0.0135, 1e-1), (0.001, 1e-1)))
    # means, stds = mcmc.sample(truth)

    mcmc = ETP_SRI_MCMC(bvalues, gaussian_noise=True, data_scale=1e-2) #, priors=((0.07, 1e-1), (0.0135, 1e-1), (0.001, 1e-1)))
    mcmc_results = mcmc.ivim_fit(signals, initial_guess=truth)
    means = mcmc_results['means']
    stds = mcmc_results['stds']

    print(f'means {means} stds {stds}')
    print(f'expected {results_reordered}')
    print(f'truth {truth}')

    signal_means = ivim_model(bvalues, means[0], means[2], means[1])
    plt.plot(bvalues, signals_noiseless, 'g', label='Noiseless Signal')
    plt.plot(bvalues, signals, '.g', label='Noisy Signal')
    plt.plot(bvalues, signal_results, 'r', label='Results Signal')
    plt.plot(bvalues, signal_means, 'b', label='Means Signal')
    plt.legend()
    # plt.show()

    mcmc.plot(overplot=truth)

    
#model1 = ETP_SRI_LinearFitting(thresholds=[200])
#model2 = IAR_LU_biexp(bounds=([0,0,0,0], [1,1,1,1]))
#model2 = IAR_LU_modified_mix()
model2 = OGC_AmsterdamUMC_biexp()

#dev_test_run(model1)
dev_test_run(model2)


def run_sampling():
    bvalues = np.array([0, 50, 100, 150, 200, 500, 800])

    def ivim_model(b, S0=1, f=0.1, Dstar=0.01, D=0.001):
        return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

    signals = ivim_model(bvalues)

    
