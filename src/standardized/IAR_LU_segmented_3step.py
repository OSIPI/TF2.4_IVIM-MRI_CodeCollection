import numpy as np
from dipy.core.gradients import gradient_table
from src.wrappers.OsipiBase import OsipiBase
from src.original.IAR_LundUniversity.ivim_fit_method_segmented_3step import IvimModelSegmented3Step


class IAR_LU_segmented_3step(OsipiBase):
    """
    Bi-exponential fitting algorithm by Ivan A. Rashid, Lund University
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Ivan A. Rashid, LU"
    id_algorithm_type = "Segmented bi-exponential fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    
    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,0] # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True
    accepted_dimensions = 1 # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, weighting=None, stats=False):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(IAR_LU_segmented_3step, self).__init__(bvalues, thresholds, bounds, initial_guess)
        
        # Check the inputs
        
        # Initialize the algorithm
        if self.bvalues is not None:
            bvec = np.zeros((self.bvalues.size, 3))
            bvec[:,2] = 1
            gtab = gradient_table(self.bvalues, bvec, b0_threshold=0)
            
            self.IAR_algorithm = IvimModelSegmented3Step(gtab)
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
            
            self.IAR_algorithm = IvimModelSegmented3Step(gtab)
            
        fit_results = self.IAR_algorithm.fit(signals)
        
        f = fit_results.model_params[1]
        Dstar = fit_results.model_params[2]
        D = fit_results.model_params[3]
        
        return f, Dstar, D