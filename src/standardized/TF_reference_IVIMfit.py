from src.wrappers.OsipiBase import OsipiBase
from src.original.TF_reference.segmented_IVIMfit import segmented_IVIM_fit
import numpy as np

class TF_reference_IVIMfit(OsipiBase):
    """
    Bi-exponential fitting algorithm by IVIM Task force
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "OSIPI IVIM TF"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D"
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

    def __init__(self, bvalues=None, thresholds=200, bounds=None, initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(TF_reference_IVIMfit, self).__init__(bvalues=bvalues, thresholds=thresholds,bounds=bounds,initial_guess=initial_guess)
        self.TF_reference_algorithm = segmented_IVIM_fit
        self.initialize(bounds, thresholds)
        self.use_initial_guess = {"f" : False, "D" : False, "Dp" : False, "S0" : False}


    def initialize(self, bounds, thresholds):
        self.use_bounds = {"f" : True, "D" : True, "Dp" : True, "S0" : True}
        if self.thresholds is None:
            self.thresholds = 200
            print('warning, no thresholds were defined, so default threshold are used of 200')
        else:
            self.thresholds = thresholds


    def ivim_fit(self, signals, bvalues=None):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        bounds = ([self.bounds["D"][0], self.bounds["f"][0], self.bounds["Dp"][0], self.bounds["S0"][0]],
                  [self.bounds["D"][1], self.bounds["f"][1], self.bounds["Dp"][1], self.bounds["S0"][1]])

        bvalues = np.array(bvalues)

        if np.any(signals < 0):
            signals = np.clip(signals,0.01, None)
            print('warning, negative values in signal: values clipped to 0.01')


        fit_results = self.TF_reference_algorithm(bvalues,signals,b_cutoff=self.thresholds, bounds=bounds)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]

        return results