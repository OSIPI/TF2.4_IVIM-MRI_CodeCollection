import numpy as np
from scipy.stats import norm
from IVIM_fitting import IVIM_seg

def ivim_signal(b, S0, f, D_star, D):
    return S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))

def diffusion_signal(b, S0, f, D):
    return S0*(1-f)*np.exp(-b*D)

def generate_noise(loc, sigma):
    real_component = norm.rvs(loc=loc, scale=sigma/loc)
    imaginary_component = norm.rvs(loc=loc, scale=sigma/loc)
    return np.absolute(complex(real_component, imaginary_component))

def add_rician_noise(signal, SNR):
    sigma = signal[-1]/SNR
    # Sample real and imaginary noise components from gaussian distributions
    # Use the last b-value as the SNR baseline in order to avoid the noise floor
    noise = np.array([generate_noise(signal_value, sigma) for signal_value in signal])
    
    # Add the two components to the signal and take the magniutde of the result
    noised_signal = signal + noise
    noised_signal = np.absolute(noised_signal)

    return noised_signal

# Ground truth
factor = 1
S0 = 1
f = 0.1
D_star = 30e-3
D = 1e-3
rescale_units = False

# Settings
lower_bounds = (0, 5, 0)
upper_bounds = (1, 100, 4)
bounds_um = np.array((lower_bounds, upper_bounds))

lower_bounds = (0, 0.005, 0)
upper_bounds = (1, 0.1, 0.004)
bounds_mm = (lower_bounds, upper_bounds)
initial_guess_mm = (1, 0.2, 0.03, 0.001)

# Create gtab containing b-value informations
bvals = np.array([0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800]).astype(float)

# Signal
signal = ivim_signal(bvals, S0, f, D_star, D)
noised_signal = add_rician_noise(signal, 3)
noised_signal /= noised_signal[0]

noised_signal6 = add_rician_noise(signal, 6)
noised_signal6 /= noised_signal6[0]



blim = 200


# IVIM_seg
pars = IVIM_seg(signal, bvals, np.array([[0, 0, 0, 0],[3e-3, np.inf, 1, 1]]), blim, False)


