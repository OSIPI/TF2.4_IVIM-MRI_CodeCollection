"""
Code to calculate IVIM maps

Segmented fitting approach
First fit D using a WLLS

"""

import numpy as np
import statsmodels.api as sm
import scipy

def ivim_biexp(bvalues, D, f, Dp, S0=1):
    return (S0 * (f * np.exp(-bvalues * Dp) + (1 - f) * np.exp(-bvalues * D)))


def segmented_IVIM_fit(bvalues, dw_data, b_cutoff = 200, bounds=([0.0001, 0.0, 0.001], [0.004, 0.7, 0.01])):
    """
        A segmented fitting implementation for a bi-exponential model.
        First D is fitted using a mono exponential model on all signal above the bvalue cutoff using an iterative WLLS
        Then f is fitted by using the b=0 intercept from the mono expontential fit and substracting this from the measured signal at b=0
        Then D* is fitted using a bi-exponential model with fixed D and f

        """
    bvalues_D = bvalues[bvalues >= b_cutoff]
    dw_data_D = dw_data[bvalues >= b_cutoff]

    D, b0_intercept = d_fit_iterative_wls(bvalues_D, np.log(dw_data_D))

    D = D
    S0 = dw_data[bvalues == 0].item()
    f = (S0 - b0_intercept) / S0

    Dp = scipy.optimize.least_squares(
    fun=lambda Dp: ivim_biexp(bvalues, D, f, Dp[0]) - dw_data,
    x0=D*10,  # Initial guess for D*
    bounds=(bounds[0][2], bounds[1][2]))

    Dp = Dp.x[0]

    return D, f, Dp


def d_fit_iterative_wls(bvalues_D, log_signal, max_iter=100, tolerance= 1e-6):
    """
    Function to calculate D using an iterative wlls on the log(signal)

    Parameters:
    log_signal: the log() of the signal above the threshold for segmented fitting

    bvalues_D: all bvalues above the threshold for fitting D

    max_iter: the maximum number of iterations for WLS

    tolerance: the tolerance for early stopping of the WLS
    """

    bvals = sm.add_constant(bvalues_D)
    weights = np.ones_like(log_signal)

    for i in range(max_iter):
        # Weighted linear regression
        model = sm.WLS(log_signal, bvals, weights=weights)
        results = model.fit()
        signal_pred = results.predict(bvals)
        residuals = log_signal - signal_pred
        new_weights = 1 / np.maximum(residuals ** 2, 1e-10)

        if np.linalg.norm(new_weights - weights) < tolerance:
            break
        weights = new_weights

    ln_b0_intercept, D_neg  = results.params
    b0_intercept = np.exp(ln_b0_intercept)
    D = -D_neg
    return D, b0_intercept