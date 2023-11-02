import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.ETP_SRI.LinearFitting import LinearFit


class ETP_SRI_LinearFitting(OsipiBase):
    """WIP
    Implementation and execution of the submitted algorithm
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Eric T. Peterson, SRI"
    id_algorithm_type = "Linear fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared"
    
    # Algorithm requirements
    required_bvalues = 3
    required_thresholds = [0,1] # Interval from 1 to 1, in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = False
    accepted_dimensions = 1
    # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, weighting=None, stats=False):
        """
            Everything this method requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(ETP_SRI_LinearFitting, self).__init__(bvalues, thresholds, bounds, initial_guess)
        
        # Could be a good idea to have all the submission-specfic variable be 
        # defined with initials?
        self.ETP_weighting = weighting
        self.ETP_stats = stats
        
        # Check the inputs
        
    
    def ivim_fit(self, signals, bvalues=None, linear_fit_option=False, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.
            linear_fit_option (bool, optional): This fit has an option to only run a linear fit. Defaults to False.

        Returns:
            _type_: _description_
        """
        if bvalues is None:
            bvalues = self.bvalues
        
        if self.thresholds is None:
            ETP_object = LinearFit()
        else:
            ETP_object = LinearFit(self.thresholds[0])
            
        if linear_fit_option:
            f, Dstar = ETP_object.linear_fit(bvalues, signals)
            return f, Dstar
        else: 
            f, D, Dstar = ETP_object.ivim_fit(bvalues, signals)
            return f, Dstar, D
    
