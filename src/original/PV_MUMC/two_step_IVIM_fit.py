"""
January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

requirements:
numpy
tqdm
scipy
"""

# load relevant libraries
from scipy.optimize import curve_fit
import numpy as np
import tqdm




def two_exp_noS0(bvalues, Dpar, Fmv, Dmv):
    """ bi-exponential IVIM function, and S0 set to 1"""
    return Fmv * np.exp(-bvalues * Dmv) + (1 - Fmv ) * np.exp(-bvalues * Dpar)
       
def two_exp(bvalues, S0, Dpar, Fmv, Dmv):
    """ bi-exponential IVIM function"""
    return S0 * (Fmv * np.exp(-bvalues * Dmv) + (1 - Fmv ) * np.exp(-bvalues * Dpar))
   


def fit_least_squares_array(bvalues, dw_data, fitS0=False, bounds=([0.9, 0.0001, 0.0, 0.0025], [1.1, 0.0025, 0.5, 0.2]), cutoff=200):
    """
    This is the LSQ implementation, in which we first estimate Dpar using a curve fit to b-values>=cutoff;
    Second, we fit the other parameters using all b-values, while fixing Dpar from step 1. This fit
    is done on an array.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param bounds: Array with fit bounds ([S0min, Dparmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fmvmax, Dmvmax]). 
    :param cutoff: cutoff b-value used in step 1 
    :return Dpar: 1D Array with Dpar in each voxel
    :return Fmv: 1D Array with Fmv in each voxel
    :return Dmv: 1D Array with Dmv in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # initialize empty arrays
    Dpar = np.zeros(len(dw_data))
    S0 = np.zeros(len(dw_data))
    Dmv = np.zeros(len(dw_data))
    Fmv = np.zeros(len(dw_data))
    for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
        # fill arrays with fit results on a per voxel base:
        Dpar[i], Fmv[i], Dmv[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], S0_output=False, fitS0=fitS0, bounds=bounds)
    return [Dpar, Fmv, Dmv, S0]


def fit_least_squares(bvalues, dw_data, S0_output=False, fitS0=False,
                      bounds=([0.9, 0.0001, 0.0, 0.0025], [1.1, 0.003, 1, 0.2]), cutoff=200):
    """
   This is the LSQ implementation, in which we first estimate Dpar using a curve fit to b-values>=cutoff;
   Second, we fit the other parameters using all b-values, while fixing Dpar from step 1. This fit
   is done on an array. It fits a single curve
    :param bvalues: 1D Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal in different voxels at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; 
    :param fitS0: Boolean determining whether to fix S0 to 1;
    :param bounds: Array with fit bounds ([S0min, Dparmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fmvmax, Dmvmax]). 
    :param cutoff: cutoff b-value used in step 1 
    :return S0: optional 1D Array with S0 in each voxel
    :return Dpar: scalar with Dpar of the specific voxel
    :return Fmv: scalar with Fmv of the specific voxel
    :return Dmv: scalar with Dmv of the specific voxel
    """
     
    #try:
    def monofit(bvalues, Dpar, Fmv):
            return (1-Fmv)*np.exp(-bvalues * Dpar)
    
    high_b = bvalues[bvalues >= cutoff]
    high_dw_data = dw_data[bvalues >= cutoff]
    boundsmonoexp = ([bounds[0][1], bounds[0][2]], [bounds[1][1], bounds[1][2]])
    params, _ = curve_fit(monofit, high_b, high_dw_data, p0=[(bounds[1][1]-bounds[0][1])/2,(bounds[1][2]-bounds[0][2])/2], bounds=boundsmonoexp)
    Dpar1 = params[0]
    if not fitS0:
        boundsupdated=([bounds[0][2] , bounds[0][3] ],
                    [bounds[1][2] , bounds[1][3] ])    
        params, _ = curve_fit(lambda b, Fmv, Dmv: two_exp_noS0(b, Dpar1, Fmv, Dmv), bvalues, dw_data, p0=[(bounds[0][2]+bounds[1][2])/2, (bounds[0][3]+bounds[1][3])/2], bounds=boundsupdated)
        Fmv, Dmv = params[0], params[1]
        #when the fraction of a compartment equals zero (or very very small), the corresponding diffusivity is non-existing (=NaN)
        #if Fmv < 1e-4:
        #    Dmv = float("NaN")
        
    else: 
        boundsupdated = ([bounds[0][0] , bounds[0][2] , bounds[0][3] ],
                    [bounds[1][0] , bounds[1][2] , bounds[1][3] ])   
        params, _ = curve_fit(lambda b, S0, Fmv, Dmv: two_exp(b, S0, Dpar1, Fmv, Dmv), bvalues, dw_data, p0=[1, (bounds[0][2]+bounds[1][2])/2, (bounds[0][3]+bounds[1][3])/2], bounds=boundsupdated)
        S0 = params[0]
        Fmv, Dmv = params[1] , params[2]
        #when the fraction of a compartment equals zero (or very very small), the corresponding diffusivity is non-existing (=NaN)
        #if Fmv < 1e-4:
        #    Dmv = float("NaN")     
            
    if S0_output:
        return Dpar1, Fmv, Dmv, S0
    else:
        return Dpar1, Fmv, Dmv


        
