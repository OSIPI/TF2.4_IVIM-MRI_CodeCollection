import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.PV_MUMC.triexp_fitting_algorithms import fit_least_squares_tri_exp, fit_NNLS


class PV_MUMC_triexp(OsipiBase):
    """
    Tri-exponential least squares fitting algorithm by Paulien Voorter, Maastricht University
    """
    
    # Some basic stuff that identifies the algorithm
    id_author = "Paulien Voorter MUMC"
    id_algorithm_type = "Tri-exponential fit"
    id_return_parameters = "Dpar, Fint, Dint, Fmv, Dmv"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    
    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0, 0] # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
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
        super(PV_MUMC_triexp, self).__init__(bvalues, None, bounds, None)
        self.PV_algorithm = fit_least_squares_tri_exp
        
    
    def ivim_fit(self, signals, bvalues=None):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
            
        fit_results = self.PV_algorithm(bvalues, signals)
        Dpar, Fint, Dint, Fmv, Dmv = fit_results
        
        return Dpar, Fint, Dint, Fmv, Dmv
