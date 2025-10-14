from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import matlab.engine


class OJ_GU_segMATLAB(OsipiBase):
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
                           1]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = False
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?

    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = False
    supported_thresholds = True

    def __init__(self, bvalues=None, thresholds=200, bounds=None, initial_guess=None, eng=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        #super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues, bounds, initial_guess, fitS0)
        super(OJ_GU_segMATLAB, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.use_initial_guess = False
        self.initialize(bounds, thresholds)
        if eng is None:
            print('initiating matlab; this may take some time. For repeated testing one could use the optional input eng as an already initiated matlab engine')
            self.eng=matlab.engine.start_matlab()
            self.keep_alive=False
        else:
            self.eng = eng
            self.keep_alive=True

    def algorithm(self, Y, b, lim, blim):
        Y = matlab.double(Y.tolist())
        b = matlab.double(b.tolist())
        lim = matlab.double(lim.tolist())
        blim = matlab.double(blim)
        results = self.eng.IVIM_seg(Y, b, lim, blim, False,nargout=3)
        (pars, mask, gof) = results
        return pars['D'], pars['f'], pars['Dstar'], pars['S0']

    def initialize(self, bounds, thresholds):
        if bounds is None:
            print('warning, no bounds were defined, so algorithm-specific default bounds are used')
            self.bounds=([1e-6, 0, 0.003, 0],[0.003, 1.0, 0.2, 5])
        else:
            self.bounds=bounds
        self.use_bounds = True
        if thresholds is None:
            self.thresholds = 200
            print('warning, no thresholds were defined, so default bounds are used of 200')
        else:
            self.thresholds = thresholds

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
                                     np.array(self.bounds)[:,[0,3,1,2]], 
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
