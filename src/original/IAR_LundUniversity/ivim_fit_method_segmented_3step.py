
""" Classes and functions for fitting ivim model """
import numpy as np
from scipy.optimize import curve_fit
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.utils.optpkg import optional_package


class IvimModelSegmented3Step(ReconstModel):

    def __init__(self, gtab, b_threshold_upper=100, b_threshold_lower=200, \
        initial_guess=None, bounds=None, rescale_units=False):
        """The 3-step segmented fit as described in the DIPY documentation.
        https://dipy.org/documentation/1.0.0./examples_built/reconst_ivim/

        Args:
            gtab (DIPY gradient table):
                Object that holds the diffusion encoding information. In this
                case, the b-values.
            b_threshold_upper (float, optional): 
                The upper b-value threshold of the perfusion fit. 
                Defaults to 100.
            b_threshold_lower (float, optional): 
                The lower b-value threshold of the diffusion fit. 
                Defaults to 200.
            initial_guess (tuple, optional): 
                The intial guess for the parameters. Defaults to None.
            bounds (tuple, optional): 
                The bounds input as a tuple of lower bounds and upper bounds,
                specified in the order f, D*, D. Defaults to None.
            rescale_units (bool, optional): 
                Set to True if the units are to be scaled to µm2/ms. Make sure
                that the b-values are already specified in these units.
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
        
        perf_bval_indices = np.where(self.bvals <= self.perf_b_threshold_upper)[0]
        perf_bvals = self.bvals[perf_bval_indices]
        perf_data = data[perf_bval_indices]
        
        S0_perf_est, D_star_est = curve_fit(self.perfusion_signal, perf_bvals, perf_data, \
            bounds=perf_bounds, p0=np.take(self.initial_guess, [0, 2]), maxfev=10000)[0]
        
        # Calculate the estimation of f based on the two S0 estimates
        f_est = S0_perf_est/(S0_perf_est + S0_diff_est)
        
        # Fit to the full bi-exponential, f estimate as initial guess, D fixed
        full_initial_guess = np.array([self.initial_guess[0], f_est, self.initial_guess[2]])
        
        full_bounds_lower = self.bounds[0][:-1]
        full_bounds_upper = self.bounds[1][:-1]
        full_bounds = (full_bounds_lower, full_bounds_upper)
        
        S0_est, f_est, D_star_est = curve_fit(lambda b, S0, f, D_star: self.ivim_signal(b, S0, f, D_star, D_est), self.bvals, data, bounds=full_bounds, p0=full_initial_guess, maxfev=10000)[0]
        
        # Set the results and rescale S0
        result = np.array([S0_est, f_est, D_star_est, D_est])
        result[0] *= data_max

        return IvimFit(self, result)

    def diffusion_signal(self, b, S0, D):
        return S0*np.exp(-b*D)
    
    def perfusion_signal(self, b, S0, D_star):
        return S0*np.exp(-b*D_star)
    
    def ivim_signal(self, b, S0, f, D_star, D):
        return S0*(f*np.exp(-b*D_star) + (1-f)*np.exp(-b*D))
    
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
