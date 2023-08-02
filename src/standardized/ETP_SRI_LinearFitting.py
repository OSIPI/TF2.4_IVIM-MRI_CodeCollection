import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.ETP_SRI.LinearFitting import LinearFit


class ETP_SRI_LinearFitting(OsipiBase):
    """WIP
    Implementation and execution of the submitted algorithm
    """
    
    def ivim_fit(self, signals, b_bvalues):
        ETP_object = LinearFit(self.thresholds[0])
        
        f, D, Dstar = ETP_object.ivim_fit(b_values, signals)
        
        
        
        return (f, Dstar, D)
    


# Simple test code... 
b_values = np.array([0, 200, 500, 800])

def ivim_model(b, S0=1, f=0.1, Dstar=0.03, D=0.001):
    return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

signals = ivim_model(b_values)

model = ETP_SRI_LinearFitting()
model.thresholds = [200]
results = model.ivim_fit(b_values, signals)