from src.wrappers.OsipiBase import OsipiBase
import numpy as np
import matlab.engine


class ASD_MemorialSloanKettering_QAMPER_IVIM(OsipiBase):
    """
    Bi-exponential fitting algorithm by Eve LoCastro and Amita Shukla-Dave, Memorial Sloan Kettering
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "LoCastro, Dr. Ramesh Paudyal, Dr. Amita Shukla-Dave"
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
    supported_thresholds = False

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, eng=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        #super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues, bounds, initial_guess, fitS0)
        super(ASD_MemorialSloanKettering_QAMPER_IVIM, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.initialize(bounds, initial_guess)
        if eng is None:
            print('initiating matlab; this may take some time. For repeated testing one could use the optional input eng as an already initiated matlab engine')
            self.eng=matlab.engine.start_matlab()
            self.keep_alive=False
        else:
            self.eng = eng
            self.keep_alive=True

    def algorithm(self,dwi_arr, bval_arr, LB0, UB0, x0in):
        dwi_arr = matlab.double(dwi_arr.tolist())
        bval_arr = matlab.double(bval_arr.tolist())
        LB0 = matlab.double(LB0.tolist())
        UB0 = matlab.double(UB0.tolist())
        x0in = matlab.double(x0in.tolist())
        results = self.eng.IVIM_standard_bcin(
            dwi_arr, bval_arr, 0.0, LB0, UB0, x0in, False, 0, 0,nargout=11)
        (f_arr, D_arr, Dx_arr, s0_arr, fitted_dwi_arr, RSS, rms_val, chi, AIC, BIC, R_sq) = results
        return D_arr/1000, f_arr, Dx_arr/1000, s0_arr

    def initialize(self, bounds, initial_guess):
        if bounds is None:
            print('warning, no bounds were defined, so algorithm-specific default bounds are used')
            self.bounds=([1e-6, 0, 0.004, 0],[0.003, 1.0, 0.2, 5])
        else:
            self.bounds=bounds
        if initial_guess is None:
            print('warning, no initial guesses were defined, so algorithm-specific default initial guess is used')
            self.initial_guess = [0.001, 0.2, 0.01, 1]
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

        bvalues=np.array(bvalues)
        LB = np.array(self.bounds[0])[[1,0,2,3]]
        UB = np.array(self.bounds[1])[[1,0,2,3]]

        fit_results = self.algorithm(np.array(signals)[:,np.newaxis], bvalues, LB, UB, np.array(self.initial_guess)[[1,0,2,3]])

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
