import numpy as np
from dipy.core.gradients import gradient_table
from scipy.stats import norm
import matplotlib.pyplot as plt
import scienceplots
import ivim_fit_method_biexp
import ivim_fit_method_subtracted
import ivim_fit_method_sivim
import ivim_fit_method_linear
import ivim_fit_method_segmented_3step
import ivim_fit_method_segmented_2step
import ivim_fit_method_modified_mix
import ivim_fit_method_modified_topopro

plt.style.use(["science", "ieee"])

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
bounds_um = (lower_bounds, upper_bounds)

lower_bounds = (0, 0.005, 0)
upper_bounds = (1, 0.1, 0.004)
bounds_mm = (lower_bounds, upper_bounds)
initial_guess_mm = (1, 0.2, 0.03, 0.001)

# Create gtab containing b-value informations
bvals = np.array([0, 50, 240, 800])/factor
bvals = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, \
    150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
bvals = np.array([0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800])
#bvals = np.array([0, 50, 240, 800])
bvec = np.zeros((bvals.size, 3))
bvec[:,2] = 1
gtab = gradient_table(bvals, bvec, b0_threshold=0)

# Signal
signal = ivim_signal(bvals, S0, f, D_star, D)
noised_signal = add_rician_noise(signal, 3)
noised_signal /= noised_signal[0]

noised_signal6 = add_rician_noise(signal, 6)
noised_signal6 /= noised_signal6[0]






# biexp fit
biexp_model = ivim_fit_method_biexp.IvimModelBiExp(gtab, bounds=bounds_mm, initial_guess=initial_guess_mm, rescale_units=rescale_units)
biexp_fit = biexp_model.fit(noised_signal)

# sIVIM fit
lower_bounds_sivim = (0, 0)
upper_bounds_sivim = (1, 4/factor)
bounds_mm_sivim = (lower_bounds_sivim, upper_bounds_sivim)
initial_guess_mm_sivim = (1, 0.2, 0.001)
sivim_model = ivim_fit_method_sivim.IvimModelsIVIM(gtab, b_threshold=0.2, bounds=bounds_mm_sivim, initial_guess=initial_guess_mm_sivim, rescale_units=rescale_units)
sivim_fit = sivim_model.fit(noised_signal)

# linear fit
linear_model = ivim_fit_method_linear.IvimModelLinear(gtab, b_threshold=0.2, bounds=bounds_mm_sivim, rescale_units=rescale_units)
linear_fit = linear_model.fit(noised_signal)

# Subtracted fit (Le Bihan 2019)
subtracted_model = ivim_fit_method_subtracted.IvimModelSubtracted(gtab, bounds=bounds_mm, initial_guess=initial_guess_mm, rescale_units=rescale_units)#, b_threshold_lower=0.2, b_threshold_upper=0.1)
subtracted_fit = subtracted_model.fit(noised_signal)

# Segmented fit (3 step) (DIPY)
segmented_3step_model = ivim_fit_method_segmented_3step.IvimModelSegmented3Step(gtab, bounds=bounds_mm, initial_guess=initial_guess_mm, rescale_units=rescale_units)#, b_threshold_lower=0.2, b_threshold_upper=0.1)
segmented_3step_fit = segmented_3step_model.fit(noised_signal)

# Segmented fit (2 step) (Conventional method)
segmented_2step_model = ivim_fit_method_segmented_2step.IvimModelSegmented2Step(gtab, bounds=bounds_mm, initial_guess=initial_guess_mm, rescale_units=rescale_units)#, b_threshold_lower=0.2)
segmented_2step_fit = segmented_2step_model.fit(noised_signal)
segmented_2step_fit6 = segmented_2step_model.fit(noised_signal6)

# MIX (Farooq et al.)
mix_model = ivim_fit_method_modified_mix.IvimModelVP(gtab, bounds=bounds_mm, rescale_units=rescale_units, rescale_results_to_mm2_s=True)
mix_fit = mix_model.fit(noised_signal)
mix_fit6 = mix_model.fit(noised_signal6)

# TopoPro (Fadnavis et al.)
topopro_model = ivim_fit_method_modified_topopro.IvimModelTopoPro(gtab, bounds=bounds_mm, rescale_units=rescale_units, rescale_results_to_mm2_s=True)
topopro_fit = topopro_model.fit(noised_signal)
topopro_fit6 = topopro_model.fit(noised_signal6)

# Print estimates
print(f"Bi-exponential: {biexp_fit.model_params}")
print(f"Linear: {linear_fit.model_params}")
print(f"sIVIM: {sivim_fit.model_params}")
print(f"Subtracted: {subtracted_fit.model_params}")
print(f"3-step segmented: {segmented_3step_fit.model_params}")
print(f"2-step segmented: {segmented_2step_fit.model_params}")
print(f"MIX: {mix_fit.model_params}")
print(f"TopoPro: {topopro_fit.model_params}")


