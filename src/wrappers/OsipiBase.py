import numpy as np
from scipy.stats import norm


class OsipiBase:
    """The base class for OSIPI IVIM fitting"""
    
    #def __init__(self, author, data_dimension, thresholds_required, guess_required, bounds_required):
    #    pass

    def fit_osipi(self, data=None, b_values=None, initial_guess=None, bounds=None, **kwargs):
        """Fits the data with the bvalues
        Returns [S0, f, D*, D]
        """
        #self.parameter_estimates = self.ivim_fit(data, b_values)
        pass

    def accepted_dimensions_osipi(self):
        """The array of accepted dimensions
        e.g.
        (1D,   2D,   3D,    4D,    5D,    6D)
        (True, True, False, False, False, False)
        """
        return (False,) * 6

    def accepts_dimension_osipi(self, dim):
        """Query if the selection dimension is fittable"""
        accepted = self.accepted_dimensions()
        if dim < 0 or dim > len(accepted):
            return False
        return accepted[dim]
    
    def check_thresholds_required_osipi(self):
        """How many segmentation thresholds does it require?"""
        if (len(self.thresholds) < self.thresholds_required[0]) or (len(self.thresholds) > self.thresholds_required[1]):
            print("Conformance error: Number of thresholds.")
            return False
        return True

    def check_guess_required_osipi():
        """Does it require an initial guess?"""
        return False

    def check_bounds_required_osipi():
        """Does it require bounds?"""
        return False
    
    def check_b_values_required_osipi():
        """Minimum number of b-values required"""
        pass

    def author_osipi():
        """Author identification"""
        return ''
    
    def simple_bias_and_RMSE_test(self, SNR, b_values, f, Dstar, D, noise_realizations=100):
        # Generate signal
        b_values = np.asarray(b_values)
        signals = f*np.exp(-b_values*Dstar) + (1-f)*np.exp(-b_values*D)
        
        f_estimates = np.zeros(noise_realizations)
        Dstar_estimates = np.zeros(noise_realizations)
        D_estimates = np.zeros(noise_realizations)
        for i in range(noise_realizations):
            # Add some noise
            sigma = signals[0]/SNR
            noised_signal = np.array([norm.rvs(signal, sigma) for signal in signals])
            
            # Perform fit with the noised signal
            f_estimates[i], Dstar_estimates[i], D_estimates[i] = self.ivim_fit(noised_signal, b_values)
            
        # Calculate bias
        f_bias = np.mean(f_estimates) - f
        Dstar_bias = np.mean(Dstar_estimates) - Dstar
        D_bias = np.mean(D_estimates) - D
            
        # Calculate RMSE
        f_RMSE = np.sqrt(np.var(f_estimates) + f_bias**2)
        Dstar_RMSE = np.sqrt(np.var(Dstar_estimates) + Dstar_bias**2)
        D_RMSE = np.sqrt(np.var(D_estimates) + D_bias**2)
            
        print(f"f bias:\t{f_bias}\nf RMSE:\t{f_RMSE}")
        print(f"Dstar bias:\t{Dstar_bias}\nDstar RMSE:\t{Dstar_RMSE}")
        print(f"D bias:\t{D_bias}\nD RMSE:\t{D_RMSE}")
            
    
