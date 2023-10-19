import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.OJ_GU.IVIM_fitting import IVIM_seg


class OJ_GU_seg(OsipiBase):
    """
    Segmented fitting algorithm by Oscar Jalnefjord, University of Gothenburg
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Oscar Jalnefjord, GU"
    id_algorithm_type = "Segmented fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared"
    
    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,0] # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = True
    required_bounds_optional = True # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = False
    accepted_dimensions = 1 # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(OJ_GU_seg, self).__init__(bvalues, thresholds, bounds)
        if bounds is None:
            self.bounds = np.array([[0, 0, 0, 0],[3e-3, np.inf, 1, 1]])

    def ivim_fit(self, signals, bvalues=None, verbose=False):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        if bvalues is None:
            bvalues = self.bvalues
        else:
            bvalues = np.asarray(bvalues)
        
        pars = IVIM_seg(signals, bvalues, self.bounds, self.thresholds[0], disp_prog=verbose)
        
        return pars['f'], pars['Dstar'], pars['D']