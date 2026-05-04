
""" Classes and functions for fitting ivim model """
import numpy as np
from scipy.optimize import curve_fit
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.utils.optpkg import optional_package
from scipy.signal import unit_impulse


class IvimModelsIVIM(ReconstModel):

    def __init__(self, gtab, b_threshold=200, bounds=None, initial_guess=None, rescale_units=False):
        """A simple nlls fit to the bi-exponential IVIM model. No segmentations
        are performed.

        Args:
            gtab (DIPY gradient table): 
            DIPY gradient table object containing
            information of the diffusion gradients, b-values, etc.
            
            bounds (array-like, optional): 
            Bounds expressed as [lower bounds, upper bounds] for S0, f, D*, and
            D respectively. Defaults to None.
            
            initial_guess (array-like, optional):
            The initial guess for the parameters. Defaults to None.
            
            rescale_units (bool, optional): 
            Set to True if parameters are to be returned in units of µm2/ms. 
            The conversion only works in one direction, from mm2/s to µm2/ms.
            Make sure the b-values in the gtab object are already in units of
            µm2/ms if this is used. Defaults to False.
        """
        
        self.b_threshold = b_threshold
        self.bvals = gtab.bvals[gtab.bvals >= self.b_threshold]
        self.bvals = np.insert(self.bvals, 0, 0)
        
        # Get the indices for the b-values that fulfils the condition.
        # Will be used to get the corresponding signals.
        b_threshold_idx = np.where(self.bvals >= self.b_threshold)[0][1]
        self.signal_indices = [0] + list(np.where(gtab.bvals >= self.b_threshold)[0])
        
        self.set_bounds(bounds) # Sets the bounds according to the requirements of the fits
        self.set_initial_guess(initial_guess) # Sets the initial guess if the fit requires it
        self.rescale_bounds_and_initial_guess(rescale_units) # Rescales the units of D* and D to µm2/ms if set to True
        

    @multi_voxel_fit
    def fit(self, data):
        # Normalize the data
        data_max = data.max()
        if data_max == 0:
            pass
        else:
            data = data / data_max
        
        # Sort out the signals from non-zero b-values < b-threshold
        ydata = data[self.signal_indices]
        
        
        # Perform the fit
        popt, pcov = curve_fit(self.sivim_model, self.bvals, ydata, p0=self.initial_guess,\
            bounds=self.bounds, maxfev=10000)
        
        # Set the results and rescale S0
        result = popt
        result[0] *= data_max

        return IvimFit(self, result)

    def sivim_model(self, b, S0, f, D):
        delta = unit_impulse(b.shape, idx=0)
        res = S0*(f*delta + (1-f)*np.exp(-b*D))
        return res
            
    def set_bounds(self, bounds):
        # Use this function for fits that uses curve_fit
        if bounds is None:
            self.bounds = np.array([(0, 0, 0), (np.inf, 1, 0.004)])
        else:
            self.bounds = np.array([bounds[0], bounds[1]])
            
    def set_initial_guess(self, initial_guess):
        if initial_guess is None:
            self.initial_guess = (1, 0.2, 0.001)
        else:
            self.initial_guess = initial_guess
            
    def rescale_bounds_and_initial_guess(self, rescale_units):
        if rescale_units:
            # Rescale the guess
            self.initial_guess = (self.initial_guess[0], self.initial_guess[1], \
                self.initial_guess[2]*1000)
            
            # Rescale the bounds
            lower_bounds = (self.bounds[0][0], self.bounds[0][1], \
                self.bounds[0][2]*1000)
            upper_bounds = (self.bounds[1][0], self.bounds[1][1], \
                self.bounds[1][2]*1000)
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

    #@property
    #def D_star(self):
        #return self.model_params[..., 2]

    @property
    def D(self):
        return self.model_params[..., 3]

    @property
    def shape(self):
        return self.model_params.shape[:-1]
