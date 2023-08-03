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
    author = "Eric T. Peterson, SRI"
    algorithm_type = "Linear fit"
    return_parameters = "f, D*, D"
    units = "seconds per milli metre squared"
    
    # Algorithm requirements
    b_values_required = 3
    thresholds_required = [1,1] # Interval from 1 to 1, in case submissions allow a custom number of thresholds
    bounds_required = False
    accepted_dimensions = 1
    
    def __init__(self, thresholds, weighting=None, stats=False):
        """
            Everything this method requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions compare the inputs with
            the requirements.
        """
        self.thresholds = thresholds
        # Could be a good idea to have all the submission-specfic variable be 
        # defined with initials?
        self.ETP_weighting = weighting
        
        # Check the inputs
        self.check_thresholds_required_osipi()
        
    
    def ivim_fit(self, signals, b_values):
        """
            We may want this function to be as simple as
            data and b-values in -> parameter estimates out.
            Everything else such as segmentation thresholds, bounds, etc. should
            be object attributes.
            
            This makes the execution of submissions easy and predictable.
            All execution stepts requires should be performed here.
        """
        ETP_object = LinearFit(self.thresholds[0])
        f, D, Dstar = ETP_object.ivim_fit(b_values, signals)
        
        
        
        return (f, Dstar, D)
    


# Simple test code... 
b_values = np.array([0, 200, 500, 800])

def ivim_model(b, S0=1, f=0.1, Dstar=0.03, D=0.001):
    return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

signals = ivim_model(b_values)

model = ETP_SRI_LinearFitting([200])
results = model.ivim_fit(signals, b_values)
test = model.simple_bias_and_RMSE_test(SNR=20, b_values=b_values, f=0.1, Dstar=0.03, D=0.001, noise_realizations=10)