from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import importlib.util
import os

class TCML_TechnionIIT_SuperIVIMDC(OsipiBase):
    """
    Wrapper for the SUPER-IVIM-DC algorithm by TCML Technion IIT.
    Source: https://github.com/TechnionComputationalMRILab/SUPER-IVIM-DC
    """
    
    ACCEPTED_DIMENSIONS = (2, 4) 
    
    def __init__(self, model_path=None, thresholds=None, bounds=None, initial_guess=None):
        super(TCML_TechnionIIT_SuperIVIMDC, self).__init__(
            thresholds=thresholds,
            bounds=bounds,
            initial_guess=initial_guess
        )
        self.model_path = model_path
        if importlib.util.find_spec("super_ivim_dc") is None:
            print("Warning: 'super-ivim-dc' package not found.")

    def osipi_fit(self, data, bvalues):
        from super_ivim_dc.infer import infer_from_signal
        
        if self.model_path is None or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path is missing: {self.model_path}")

        if data.ndim == 1:
            data = data[np.newaxis, :]
            
        Dp, Dt, Fp, S0 = infer_from_signal(signal=data, bvalues=bvalues, model_path=self.model_path)
        return Fp, Dp, Dt, S0