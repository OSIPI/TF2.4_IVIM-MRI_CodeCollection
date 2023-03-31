import torch


def ivim_parameters_to_signal(D, Dp, f, S0, bvalues):
    """
    converts ivim parameters to predicted signal at specified bvalues
    Args:
        D: diffusion coefficient
        Dp: pseudo diffusion coefficient
        f: perfusion fraction
        S0: signal at b=0
        bvalues: b-values (measures of diffusion weighting)

    Returns:
        relative signal: relative (when S0 = 1) signal at specified b-values
        signal: signal at specified b-values

    """
    # Calculate signal based on estimated parameters
    relative_signal = f * torch.exp(-bvalues * Dp) + (1 - f) * torch.exp(-bvalues * D)
    signal = S0 * relative_signal
    return signal
