from src.wrappers.OsipiBase import OsipiBase
from src.original.OGC_AmsterdamUMC.LSQ_fitting import fit_segmented
import numpy as np

class OGC_AmsterdamUMC_biexp_segmented(OsipiBase):
    """
    Segmented bi-exponential fitting algorithm by Oliver Gurney-Champion, Amsterdam UMC
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Oliver Gurney Champion, Amsterdam UMC"
    id_algorithm_type = "Segmented bi-exponential fit"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [1,1]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?


    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = True
    supported_thresholds = True

    def __init__(self, bvalues=None, thresholds=150, bounds=None, initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(OGC_AmsterdamUMC_biexp_segmented, self).__init__(bvalues, thresholds, bounds, initial_guess)
        self.OGC_algorithm = fit_segmented
        self.initialize(bounds, initial_guess, thresholds)
        

    def initialize(self, bounds, initial_guess, thresholds):
        if bounds is None:
            print('warning, no bounds were defined, so default bounds are used of [0, 0, 0.005, 0.7],[0.005, 1.0, 0.2, 1.3]')
            self.bounds=([0, 0, 0.005, 0.7],[0.005, 1.0, 0.2, 1.3])
        else:
            self.bounds=bounds
        if initial_guess is None:
            print('warning, no initial guesses were defined, so default bounds are used of  [0.001, 0.001, 0.01, 1]')
            self.initial_guess = [0.001, 0.001, 0.01, 1]
        else:
            self.initial_guess = initial_guess
        self.use_initial_guess = True
        self.use_bounds = True
        if thresholds is None:
            self.thresholds = 150
            print('warning, no thresholds were defined, so default bounds are used of  150')
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
        
        if self.initial_guess is not None and len(self.initial_guess) == 4:
            self.initial_guess = self.initial_guess
        bvalues=np.array(bvalues)
        fit_results = self.OGC_algorithm(bvalues, signals, bounds=self.bounds, cutoff=self.thresholds, p0=self.initial_guess)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]
        results["S0"] = fit_results[3]

        return results