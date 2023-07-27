import torch
import numpy as np

from utils.ivim.forward_model import ivim_parameters_to_signal


def simulate_ivim_signal(D, Dp, f, S0, bvalues, SNR_array, rg):
    """
    simulate ivim signal

    Args:
        D: diffusion coefficient
        Dp: pseudo diffusion coefficient
        f: perfusion fraction
        S0: signal without diffusion weighting
        bvalues: b-values (measure of diffusion weighting)
        SNR_array: noise to be added to the simulated data
        rg: random number generator

    Returns:
        simulated_data: simulated ivim signal

    """
    bvalues.sort()
    b0_bool = np.array(bvalues) == 0
    simulated_data = ivim_parameters_to_signal(torch.tensor(D), torch.tensor(Dp), torch.tensor(f), torch.tensor(S0),
                                               torch.tensor(np.asarray(bvalues)))
    simulated_data = simulated_data.cpu().detach().numpy()

    # create 2 signal arrays filled with gaussian noise
    noise_real = rg.normal(0, 1 / SNR, (1, len(bvalues)))
    noise_imag = rg.normal(0, 1 / SNR, (1, len(bvalues)))

    # add Rician noise to the simulated data
    simulated_data = np.sqrt(np.power(simulated_data + noise_real, 2) + np.power(noise_imag, 2)).squeeze()

    # renormalize simulated data to noisy S0
    S0_noisy = np.mean(simulated_data[b0_bool])
    simulated_data /= S0_noisy
    return simulated_data

