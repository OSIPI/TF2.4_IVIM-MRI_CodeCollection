"""
Created on Aug 30 2022

Script to demonstrate usage of standalone DWI functions:
Generate_ADC
Generate_IVIMmaps

@authors: s.zijlema, p.vhoudt, k.baas, e.kooreman
Group: Uulke van der Heide - NKI 
"""

import numpy as np
from DWI_functions_standalone import generate_ADC_standalone, generate_IVIMmaps_standalone
import matplotlib.pyplot as plt
import pathlib

#Load some DWI data
file=pathlib.Path(__file__)
file_path = file.with_name('IVIM_b0_15_150_500.npy').as_posix()
DWI_data = np.load(file_path)

# Specify b values that are in the data
bvalues = [0, 15, 150, 500]

# Specifiy if you want to plot the results
plot_results = False

# Generate ADC map and define the min and max b value that will be used for ADC fitting
ADClog, b0_intercept, used_bvalues = generate_ADC_standalone(DWI_data,bvalues, bmin=150, bmax=500)

# Generate ADC and define the min and max b value that will be used for ADC fitting as well as the max b value
# that will be used for separate Dstar fitting
ADClog, perfFrac, Dstar = generate_IVIMmaps_standalone(DWI_data, bvalues, bminADC=150, bmaxADC=500, bmaxDstar=150)
Slice = 5

if plot_results == True:
    # Plot the ADC, f and D* maps
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(ADClog[:,Slice,:], cmap='viridis', vmin=0, vmax=5)
    axes[0].set_title("ADC")
    axes[0].axis('off')

    axes[1].imshow(perfFrac[:,Slice,:], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("f map")
    axes[1].axis('off')

    axes[2].imshow(Dstar[:,Slice,:], cmap='viridis', vmin=0, vmax=200)
    axes[2].set_title("D*")
    axes[2].axis('off')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

print("finish")


