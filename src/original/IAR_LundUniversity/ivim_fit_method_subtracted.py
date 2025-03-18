
""" Classes and functions for fitting ivim model """
import numpy as np
from scipy.optimize import curve_fit
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.utils.optpkg import optional_package


class IvimModelSubtracted(ReconstModel):

    def __init__(self, gtab, b_threshold_upper=100, b_threshold_lower=200, \
        initial_guess=None, bounds=None, rescale_units=False):
        """The subtracted method described by Le Bihan in
        What can we see with IVIM MRI? NeuroImage. 2019 Feb 15;187:56–67. 

        Args:
            gtab (DIPY gtab class): 
                Object that holds the b-values.
            b_threshold_upper (int, optional): 
                The upper threshold for the D* fit. Defaults to 100.
            b_threshold_lower (int, optional): 
                The lower threshold of the D fit. Defaults to 200.
            initial_guess (array-like, optional): 
                Initial guesses for f, D*, D repsectively. Defaults to None.
            bounds (array-like, optional): 
                Tupple of (lower bounds, upper bounds) for f, D*, D respectively. 
                Defaults to None.
            rescale_units (bool, optional): 
                Rescales the guesses and bounds to units of um2/ms. Make sure
                the b-values are already in these units if used. 
                Defaults to False.
        """
        
        
        self.bvals = gtab.bvals
        self.perf_b_threshold_upper = b_threshold_upper
        self.diff_b_threshold_lower = b_threshold_lower
        
        self.set_bounds(bounds)
        self.set_initial_guess(initial_guess)
        self.rescale_bounds_and_initial_guess(rescale_units)
        

    @multi_voxel_fit
    def fit(self, data):
        # Normalize the data
        data_max = data.max()
        if data_max == 0:
            pass
        else:
            data = data / data_max
        
        ### Fit the diffusion signal to bvals >= diff_b_threshold_lower
        diff_bounds = [(self.bounds[0][0], self.bounds[0][3]), \
            (self.bounds[1][0], self.bounds[1][3])] # Bounds for S0 and D
        
        diff_bval_indices = np.where(self.bvals >= self.diff_b_threshold_lower)[0]
        diff_bvals = self.bvals[diff_bval_indices]
        diff_data = data[diff_bval_indices]
        
        S0_diff_est, D_est = curve_fit(self.diffusion_signal, diff_bvals, diff_data, \
            bounds=diff_bounds, p0=np.take(self.initial_guess, [0, 3]), maxfev=10000)[0]
        
        
        ### Fit the perfusion signal to bvals <= perf_b_threshold_upper
        perf_bounds = [(self.bounds[0][0], self.bounds[0][2]), \
            (self.bounds[1][0], self.bounds[1][2])] # Bounds for S0 and D*
        
        perf_bvals = self.bvals[self.bvals <= self.perf_b_threshold_upper]
        diff_data_to_be_removed = self.diffusion_signal(perf_bvals, S0_diff_est, D_est)
        perf_bval_indices = np.where(self.bvals <= self.perf_b_threshold_upper)[0]
        perf_bvals = self.bvals[perf_bval_indices]
        perf_data = data[perf_bval_indices] - diff_data_to_be_removed # Subtract the diffusion signal from the total to get the perfusion signal
        
        S0_perf_est, D_star_est = curve_fit(self.perfusion_signal, perf_bvals, perf_data, \
            bounds=perf_bounds, p0=np.take(self.initial_guess, [0, 2]), maxfev=10000)[0]
        
        # Calculate the estimation of f based on the two S0 estimates
        f_est = S0_perf_est/(S0_perf_est + S0_diff_est)
        
        # Set the results and rescale S0
        result = np.array([S0_perf_est+S0_diff_est, f_est, D_star_est, D_est])
        result[0] *= data_max

        return IvimFit(self, result)

    def diffusion_signal(self, b, S0, D):
        return S0*np.exp(-b*D)
    
    def perfusion_signal(self, b, S0, D_star):
        return S0*np.exp(-b*D_star)
    
    def set_bounds(self, bounds):
        # Use this function for fits that uses curve_fit
        if bounds is None:
            self.bounds = np.array([(0, 0, 0.005, 0), (np.inf, 1, 0.1, 0.004)])
        else:
            self.bounds = np.array([bounds[0], bounds[1]])
            
    def set_initial_guess(self, initial_guess):
        if initial_guess is None:
            self.initial_guess = (1, 0.2, 0.03, 0.001)
        else:
            self.initial_guess = initial_guess
            
    def rescale_bounds_and_initial_guess(self, rescale_units):
        if rescale_units:
            # Rescale the guess
            self.initial_guess = (self.initial_guess[0], self.initial_guess[1], \
                self.initial_guess[2]*1000, self.initial_guess[3]*1000)
            
            # Rescale the bounds
            lower_bounds = (self.bounds[0][0], self.bounds[0][1], \
                self.bounds[0][2]*1000, self.bounds[0][3]*1000)
            upper_bounds = (self.bounds[1][0], self.bounds[1][1], \
                self.bounds[1][2]*1000, self.bounds[1][3]*1000)
            self.bounds = (lower_bounds, upper_bounds)

class IvimFit(object):

    def __init__(self, model, model_params):
        """ Initialize a IvimFit class instance.
            Parameters
            ----------
            model : Model class
            model_params : array
            The parameters of the model. In this case it is an
            array of ivim parameters. If the fitting is done
            for multi_voxel data, the multi_voxel decorator will
            run the fitting on all the voxels and model_params
            will be an array of the dimensions (data[:-1], 4),
            i.e., there will be 4 parameters for each of the voxels.
        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def S0_predicted(self):
        return self.model_params[..., 0]

    @property
    def perfusion_fraction(self):
        return self.model_params[..., 1]

    @property
    def D_star(self):
        return self.model_params[..., 2]

    @property
    def D(self):
        return self.model_params[..., 3]

    @property
    def shape(self):
        return self.model_params.shape[:-1]
