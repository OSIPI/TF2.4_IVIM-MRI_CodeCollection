from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def ivim_signal(b, S0, f, D_star, D):
    return S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))

def diffusion_signal(b, S0, f, D):
    return S0*(1-f)*np.exp(-b*D)

def generate_noise(loc, sigma):
    #real_component = norm.rvs(loc=loc, scale=sigma)
    #imaginary_component = norm.rvs(loc=loc, scale=sigma)
    #return complex(real_component, imaginary_component)
    noise = norm.rvs(loc=loc, scale=sigma)
    return noise

def add_rician_noise(signal, SNR):
    sigma = signal[-1]/SNR
    # Sample real and imaginary noise components from gaussian distributions
    # Use the last b-value as the SNR baseline in order to avoid the noise floor
    noise = np.array([generate_noise(signal_value, sigma) for signal_value in signal])
    
    # Add the two components to the signal and take the magniutde of the result
    #noised_signal = signal + noise
    noised_signal = noise
    noised_signal = np.absolute(noised_signal)

    return noised_signal

def add_gaussian_noise(signal, SNR):
    sigma = signal[0]/SNR
    noise = np.array([generate_noise(signal_value, sigma) for signal_value in signal])
    noised_signal = noise
    return noised_signal

# Simulation parameters
S0 = 1
f = 0.1
Dstar = 30e-3
D = 1e-3

# Algorithms
# Would be nice with a Bayesian example as well 
algorithms = ["IAR_LU_biexp", # Biexp NLLS
              "IAR_LU_segmented_2step", # Segmented
              "IAR_LU_segmented_3step", # Segmented
              "IAR_LU_subtracted", # Segmented
              "OJ_GU_seg", # Segmented
              "IAR_LU_modified_mix", # Variable projection
              "IAR_LU_modified_topopro"] # Variable projection

legend_labels = ["Bi-exponential NLLS",
                 "Segmented 1",
                 "Segmented 2",
                 "Segmented 3",
                 "Segmented 4",
                 "Variable projection 1",
                 "Variable projection 2"]

# Generate signal
#noised_signal /= noised_signal[0]

# Ground truth signal
bvalues_full = np.linspace(0,800,1000)
unnoised_signal_full = ivim_signal(bvalues_full, S0, f, Dstar, D)


def plot_algorithm_estimate(algorithm, ax, legend_label, noised_signal, bvalues):
    # Define algorithm and get estimates
    algorithm = OsipiBase(algorithm=algorithm)
    f_est, Dstar_est, D_est = algorithm.osipi_fit(noised_signal, bvalues)

    # Get the estimated signal
    est_signal = ivim_signal(bvalues_full, S0, f_est, Dstar_est, D_est)

    # Plot estimated signal
    ax.plot(bvalues_full, est_signal, label=legend_label)

def simulate_and_plot_for_SNR(SNR):
    # Create figure and plot ground truth and noised signals
    fig, ax = plt.subplots()
    ax.plot(bvalues_full, unnoised_signal_full, ls="--", marker="", color="black", label="Ground truth")
    ax.set_title(f"SNR {SNR} at b-value 0 s/mm$^2$")
    
    # Generate and plot simulated measurements
    bvalues = np.array([0,20,40,60,80,100,150,200,300,400,500,600,700,800])
    unnoised_signal = ivim_signal(bvalues, S0, f, Dstar, D)
    noised_signal = add_gaussian_noise(unnoised_signal, SNR)
    
    ax.plot(bvalues, noised_signal, ls="", marker=".", color="red")
    
    for i in range(len(algorithms)):
        plot_algorithm_estimate(algorithms[i], ax=ax, legend_label=legend_labels[i], noised_signal=noised_signal, bvalues=bvalues)
    
    fig.legend(frameon=False, bbox_to_anchor=(1.25, 0.75))
        
simulate_and_plot_for_SNR(10)