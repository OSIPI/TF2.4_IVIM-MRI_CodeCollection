import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.PV_MUMC.two_step_IVIM_fit import fit_least_squares_array, fit_least_squares


class PV_MUMC_biexp(OsipiBase):
    """
    Bi-exponential fitting algorithm by Paulien Voorter, Maastricht University
    """
    
    # Some basic stuff that identifies the algorithm
    id_author = "Paulien Voorter MUMC"
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
    accepted_dimensions = 1 # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, weighting=None, stats=False):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(PV_MUMC_biexp, self).__init__(bvalues, None, bounds, None)
        self.PV_algorithm = fit_least_squares
        
    
    def ivim_fit(self, signals, bvalues=None):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        
            
        fit_results = self.PV_algorithm(bvalues, signals)
        
        f = fit_results[1]
        Dstar = fit_results[2]
        D = fit_results[0]
        
        return f, Dstar, D
