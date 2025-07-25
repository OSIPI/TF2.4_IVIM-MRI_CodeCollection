import numpy as np
from dipy.core.gradients import gradient_table
from src.wrappers.OsipiBase import OsipiBase
from src.original.IAR_LundUniversity.ivim_fit_method_modified_topopro import IvimModelTopoPro


class IAR_LU_modified_topopro(OsipiBase):
    """
    Bi-exponential fitting algorithm by Ivan A. Rashid, Lund University
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Ivan A. Rashid, LU"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    
    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,0] # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True
    
    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = False
    supported_thresholds = False
    supported_dimensions = 1
    supported_priors = False
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, weighting=None, stats=False):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(IAR_LU_modified_topopro, self).__init__(bvalues, thresholds, bounds, initial_guess)
        if bounds is not None:
            print('warning, bounds from wrapper are not (yet) used in this algorithm')
        self.use_bounds = False
        self.use_initial_guess = False
        # Check the inputs
        
        # Initialize the algorithm
        if self.bvalues is not None:
            bvec = np.zeros((self.bvalues.size, 3))
            bvec[:,2] = 1
            gtab = gradient_table(self.bvalues, bvec, b0_threshold=0)
            
            self.IAR_algorithm = IvimModelTopoPro(gtab, bounds=self.bounds, rescale_results_to_mm2_s=True)
        else:
            self.IAR_algorithm = None
        
    
    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        
        if self.IAR_algorithm is None:
            if bvalues is None:
                bvalues = self.bvalues
            else:
                bvalues = np.asarray(bvalues)
            
            bvec = np.zeros((bvalues.size, 3))
            bvec[:,2] = 1
            gtab = gradient_table(bvalues, bvec, b0_threshold=0)
            
            self.IAR_algorithm = IvimModelTopoPro(gtab, bounds=self.bounds, rescale_results_to_mm2_s=True)
            
        fit_results = self.IAR_algorithm.fit(signals)
        
        #f = fit_results.model_params[1]
        #Dstar = fit_results.model_params[2]
        #D = fit_results.model_params[3]
        
        #return f, Dstar, D
        results = {}
        results["f"] = fit_results.model_params[1]
        results["Dp"] = fit_results.model_params[2]
        results["D"] = fit_results.model_params[3]
        
        return results