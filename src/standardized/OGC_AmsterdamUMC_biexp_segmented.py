from src.wrappers.OsipiBase import OsipiBase
from src.original.OGC_AmsterdamUMC.LSQ_fitting import fit_segmented, fit_segmented_array
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
    required_thresholds = [1,
                           1]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True


    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = True
    supported_thresholds = True
    supported_dimensions = 1
    supported_priors = False

    def __init__(self, bvalues=None, thresholds=150, bounds=None, initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(OGC_AmsterdamUMC_biexp_segmented, self).__init__(bvalues, thresholds, bounds, initial_guess)
        self.OGC_algorithm = fit_segmented
        self.OGC_algorithm_array = fit_segmented_array
        self.initialize(thresholds)

    def initialize(self, thresholds):
        self.use_initial_guess = {"f" : True, "D" : True, "Dp" : True, "S0" : True}
        self.use_bounds = {"f" : True, "D" : True, "Dp" : True, "S0" : True}

        if self.thresholds is None:
            self.thresholds = 150
            print('warning, no thresholds were defined, so default threshold of 150 was used')
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
        bounds = ([self.bounds["D"][0], self.bounds["f"][0], self.bounds["Dp"][0], self.bounds["S0"][0]],
                  [self.bounds["D"][1], self.bounds["f"][1], self.bounds["Dp"][1], self.bounds["S0"][1]])

        initial_guess = [self.initial_guess["D"], self.initial_guess["f"], self.initial_guess["Dp"], self.initial_guess["S0"]]

        bvalues=np.array(bvalues)
        fit_results = self.OGC_algorithm(bvalues, signals, bounds=bounds, cutoff=self.thresholds, p0=initial_guess)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]
        results = self.D_and_Ds_swap(results)

        return results