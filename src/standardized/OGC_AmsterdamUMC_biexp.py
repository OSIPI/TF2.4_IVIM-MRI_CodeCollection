from src.wrappers.OsipiBase import OsipiBase
from src.original.OGC_AmsterdamUMC.LSQ_fitting import fit_least_squares, fit_least_squares_array
import numpy as np

class OGC_AmsterdamUMC_biexp(OsipiBase):
    """
    Bi-exponential fitting algorithm by Oliver Gurney-Champion, Amsterdam UMC
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Oliver Gurney Champion, Amsterdam UMC"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    id_ref = "reference method in https://doi.org/10.1002/mrm.28852"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,
                           0]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True


    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = True
    supported_thresholds = False
    supported_dimensions = 1
    supported_priors = False

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        #super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues, bounds, initial_guess, fitS0)
        super(OGC_AmsterdamUMC_biexp, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.OGC_algorithm = fit_least_squares
        self.OGC_algorithm_array = fit_least_squares_array
        self.fitS0=fitS0
        self.initialize(bounds, initial_guess, fitS0)

    def initialize(self, bounds, initial_guess, fitS0):
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
        self.fitS0=fitS0
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
        fit_results = self.OGC_algorithm(bvalues, signals, p0=self.initial_guess, bounds=self.bounds, fitS0=self.fitS0)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]

        return results