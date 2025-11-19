from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import matlab.engine


class OJ_GU_bayesMATLAB(OsipiBase):
    """
    Bi-exponential fitting algorithm by Oscar Jalnefjord, University of Gothenburg
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Dr. Oscar Jalnefjord"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,
                           0]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?

    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = True
    supported_thresholds = True

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, eng=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        #super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues, bounds, initial_guess, fitS0)
        super(OJ_GU_bayesMATLAB, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.initialize(bounds, initial_guess)
        if eng is None:
            print('initiating matlab; this may take some time. For repeated testing one could use the optional input eng as an already initiated matlab engine')
            self.eng=matlab.engine.start_matlab()
            self.keep_alive=False
        else:
            self.eng = eng
            self.keep_alive=True

    def algorithm(self, Y, b, lim, blim):
        Y = matlab.double(Y.tolist())
        f = matlab.double(self.initial_guess[1])
        D = matlab.double(self.initial_guess[0])
        Dstar = matlab.double(self.initial_guess[2])
        S0 = matlab.double(self.initial_guess[3])
        b = matlab.double(b.tolist())
        lim = matlab.double(lim.tolist())
        self.eng.addpath(r'C:\TF_IVIM_OSIPI\TF2.4_IVIM-MRI_CodeCollection\src\original\OJ_GU', nargout=0)
        out = self.eng.IVIM_bayes(Y, f, D, Dstar, S0, b, lim, nargout=1)
        return out['D']['mode'], out['f']['mode'], out['Dstar']['mode'], out['S0']['mode']

    def initialize(self, bounds,initial_guess):
        if bounds is None:
            print('warning, no bounds were defined, so default bounds are used of [0, 0, 0.005, 0.7],[0.005, 1.0, 0.2, 1.3]')
            self.bounds=([0, 0, 0.005, 0.7],[0.005, 1.0, 0.2, 1.3])
        else:
            self.bounds=bounds
        if initial_guess is None:
            print('warning, no initial guesses were defined, so default bounds are used of  [0.001, 0.001, 0.01, 1]')
            self.initial_guess = [0.001, 0.1, 0.01, 1]
        else:
            self.initial_guess = initial_guess
            self.use_initial_guess = True
        self.use_initial_guess = True
        self.use_bounds = True

    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """

        fit_results = self.algorithm(np.array(signals)[:,np.newaxis], 
                                     np.array(bvalues), 
                                     np.array(self.bounds)[:,[1,0,2,3]], 
                                     self.thresholds)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]

        return results

    def clean(self):
        if not self.keep_alive:
            if hasattr(self, "eng") and self.eng:
                try:
                    self.eng.quit()
                except Exception as e:
                    print(f"Warning: Failed to quit MATLAB engine cleanly: {e}")

    def __del__(self):
        self.clean()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clean()
