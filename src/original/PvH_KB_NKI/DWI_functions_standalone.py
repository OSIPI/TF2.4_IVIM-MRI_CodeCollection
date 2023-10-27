# -*- coding: utf-8 -*-
"""
DWI_functions.py
====================================
This module can be used to calculate ADC/IVIM maps.
IVIM fitting is done in a segmented fashion: https://doi.org/10.3389/fonc.2021.705964

@authors: s.zijlema, p.vhoudt, k.baas, e.kooreman
Group: Uulke van der Heide - NKI
"""

import numpy as np

def generate_ADC_standalone(DWIdata, bvalues, bmin: int = 150, bmax: int = 1000, specificBvals: list = []):
    """
    Function to create an ADC map from DWI data.
    Required input is a numpy array with the data and the corresponding b-values

    Example use:
    generate_ADC(data, bvalues)

    Parameters
    ---------
    DWIdata: numpy array
        DWI data
    bvalues: list
        bvalues corresponding to the data
    bmin/bmax (optional)
        Used to select range of b-values that you want to include to calculate the ADC.
        Not used when specificBvals is defined.
    specificBvals (optional)
        List of b-values that should be used for ADC map calculation.
        If set, bmin/bmax options are not used.

    Returns
    -------
    ADClog
        ADC map
    b0_intercept
        Map with y-intercepts (b=0) of ADC fit
    bvalues
        b-values that matched the criteria
    """

    #Obtain indices of matching b-values
    if specificBvals == []:
        #Select only b-values >=bmin and <=bmax
        bvalueindex = [i for i,v in enumerate(bvalues) if v >= bmin and v <= bmax]
    else: #manual b-value selection
        #Check if all specified b-values were acquired
        if not all(elem in bvalues for elem in specificBvals):
            raise Exception('Not all specified b-values were found. Available b-values: '+str(bvalues)+'. You requested: '+str(specificBvals))
        #Get b-value indices
        bvalueindex = [i for i,v in enumerate(bvalues) if v in specificBvals]

    use_bvalues = [bvalues[index] for index in bvalueindex]
    if len(use_bvalues) < 2:
        raise Exception('Less than 2 b values were found that fulfill your min and max criteria.')

    # Calculate ADC map using linear regression
    ADClog, b0_intercept = fit_ADCmap_loglinear_standalone(DWIdata, bvalues, use_bvalues)
    ADClog[ADClog<0] = 0 #set values below zero to zero

    return ADClog, b0_intercept, use_bvalues

def generate_IVIMmaps_standalone(DWIdata, bvalues, bminADC=150, bmaxADC=1000, bmaxDstar=50):
    """
    Function to calculate IVIM maps (perfusion fraction and D*).
    NOTE: D* is calculated using only the lowest two b-values.

    Example use:
    generate_IVIMmaps(data, bvalues)

    Parameters
    ---------
    DWIdata: numpy array
        DWI data
    bvalues: list
        bvalues corresponding to the data
    bminADC/bmaxADC (optional)
        Used to select the range of b-values for ADC calculation.
    bmaxDstar (optional)
        Used to select the b-values for D* calculation.
    outputOptions
        ['ADC','f','D*'] (default) will output the ADC, f, and D*.


    """

    # Calculate ADC map
    ADClog, b0_intercept, used_bvalues = generate_ADC_standalone(DWIdata, bvalues, bminADC, bmaxADC)

    # Check if b=0 is acquired, otherwise f and D* cannot be calculated
    if not 0 in bvalues:  # if b=0 is not acquired
        print('B=0 is not available. Hence, only the ADC is calculated.')
        return  # stop execution

    # Get b0 data
    b0index = bvalues.index(0)  # gets first index of b=0
    DWIb0 = DWIdata[:, :, :, b0index]  # b=0 data

    #### Calculate perfusion fraction
    with np.errstate(divide='ignore', invalid='ignore'):  # hide division by 0 warnings
        perfFrac = (DWIb0 - np.exp(b0_intercept)) / DWIb0  # perfusion fraction
    perfFrac[np.isnan(perfFrac)] = 0
    perfFrac[np.isinf(perfFrac)] = 0
    perfFrac[perfFrac < 0] = 0  # f <0 is not possible
    perfFrac[perfFrac > 1] = 1  # f >1 is also not possible

    # Get matching b-values
    try:
        _, bvalues = select_b_values_standalone(bvalues, bmin=0, bmax=bmaxDstar)

        # Check if b=0 was acquired
        if 0 not in bvalues:
            raise Exception('b=0 was not acquired')

        if len(bvalues) > 2:
            print('More than two b-values were detected for D* calculation. ' +
                  'Note that only the first two will be used for D*.')

    except Exception as e:
        print("Could not calculate D* due to an error: " + str(e))
        return

    # Use b=0 and the lowest b-value to calculate D*
    with np.errstate(divide='ignore', invalid='ignore'):  # hide division by 0 warnings
        Dstar = -np.log(
            (DWIdata[:,:,:,1] / DWIdata[:,:,:,0] - (1 - perfFrac) * np.exp(-bvalues[1] * 0.001 * ADClog)) / perfFrac) / \
                bvalues[1]  # Calculate D* using f, d, and the lowest 2 b-values
    Dstar[np.isnan(Dstar)] = 0
    Dstar[np.isinf(Dstar)] = 0
    Dstar = Dstar * 1000  # scale to same as ADC

    return ADClog, perfFrac, Dstar


def select_b_values_standalone(all_bvalues, bmin=0, bmax=float("inf"), specificBvals: list = []):
    """
    Find b-values and their indices between bmin and bmax of a scan (bmin <= b <= b max)
    or of b-values that are defined in specificBvals.
    If specificBvals is defined, only b-values are returned that are acquired
    in the scan AND are present in specificBvals.

    Parameters
    ----------
    all_bvalues : list of int
        All b values that are in the data in the order that they are stored
    bmin : int
        Minimal b-value.
    bmax : int
        Maximal b-value.
    specificBvals : list
        List of b-values.


    Returns
    -------
    bvalueindex : list of int
        Indices of b-values between bmin and bmax.
    bvalues : list of int
        B-values between bmin and bmax.

    """

    # Obtain indices of matching b-values
    if specificBvals == []:
        # Select only b-values >=bmin and <=bmax
        bvalueindex = [i for i, v in enumerate(all_bvalues) if v >= bmin and v <= bmax]
    else:  # manual b-value selection
        # Check if all specified b-values were acquired
        if not all(elem in all_bvalues for elem in specificBvals):
            raise Exception('Not all specified b-values were found. Available b-values: ' + str(
                all_bvalues) + '. You requested: ' + str(specificBvals))

        # Get b-value indices
        bvalueindex = [i for i, v in enumerate(all_bvalues) if v in specificBvals]

    # Check if enough b-values were found
    if len(bvalueindex) < 2:
        raise Exception("No(t enough) b-values matched the selected range between " + str(bmin) + " and " + str(
            bmax) + " (found: " + str(len(bvalueindex)) + ")")

    # Select matching b-values
    bvalues = [all_bvalues[i] for i in bvalueindex]

    # Return index and b-values
    return bvalueindex, bvalues


def fit_ADCmap_loglinear_standalone(DWIdata, all_bvalues, use_bvalues, size=[]):
    """
    Function to calculate ADC based on linear regression.
    Taken from ADC_functions.py but adjusted to also output intercept and
    removed size input requirement (could be extracted from si_array).
    The ADC values have been validated with the Philips ADC maps (2022).

    Parameters
    ---------
    DWIdata
        DWI data
    all_bvalues
        list or array of b-values that are stored in the data
    use_bvalues
        list or array of selected b-values selected for calculation.

    Returns
    -------
        ADC calculated ADC and the intercept with the y-axis at b=0 in the same
        x,y,z coordinates as the original DWI image (i.e. size)

    """

    nr_bvalues = len(use_bvalues)

    b_mat = np.ones([nr_bvalues, 2])
    for curBvalNr in range(nr_bvalues):
        b_mat[curBvalNr, 0] = use_bvalues[curBvalNr]

    if size == []:
        size = DWIdata.shape[:-1] # b-values are stored in the last dimension

    # Extract the b values that are used for calculation
    indices = [all_bvalues.index(value) for value in use_bvalues]
    DWIdata = DWIdata[:,:,:,indices]
    DWIdata = DWIdata.transpose(3, 0, 1, 2) # code below expects different format

    with np.errstate(divide='ignore', invalid='ignore'):  # hide division by 0 warnings
        log_si = -np.log(DWIdata.reshape(nr_bvalues, -1))  # calculate log of data and reshape to one row per b-value
        #log_si = -np.log(DWIdata.reshape(-1, nr_bvalues))
    log_si[np.isinf(log_si)] = 0  # to fix 0 values in si_array

    ADC, intercept = np.linalg.lstsq(b_mat, log_si, rcond=None)[0]  # calculate ADC and b=0 intercept
    ADC = ADC.reshape(*size) * 1000  # Convert ADC to 10^-3 mm^2/s
    b0_intercept = -intercept.reshape(*size)  # can be used to calculate perfusion fraction

    return ADC, b0_intercept


