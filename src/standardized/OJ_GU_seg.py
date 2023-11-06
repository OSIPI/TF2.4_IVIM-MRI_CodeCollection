import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.OJ_GU.ivim_seg import seg

class OJ_GU_seg(OsipiBase):
    """
    Segmented fitting algorithm by Oscar Jalnefjord, University of Gothenburg
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Oscar Jalnefjord, GU"
    id_algorithm_type = "Segmented bi-exponential fit"
    id_return_parameters = "f, D*, D"
    id_units = "mm2/s"
    
    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,0] # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = False # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = False
    accepted_dimensions = 1 # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, weighting=None, stats=False):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(OJ_GU_seg, self).__init__(bvalues, thresholds, bounds, initial_guess)
        
        # Check the inputs
        
        # Initialize the algorithm
        
    
    def ivim_fit(self, signals, bvalues=None):
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

        if self.thresholds is None:
            bthr = 200
        else:
            bthr = self.thresholds[0]
                    
        fit_results = seg(signals, bvalues, bthr)
        
        f = np.squeeze(fit_results['f'])
        Dstar = np.squeeze(fit_results['Dstar'])
        D = np.squeeze(fit_results['D'])
        
        return f, Dstar, D