from src.wrappers.OsipiBase import OsipiBase
from super_ivim_dc.source.Classsic_ivim_fit import fit_least_squares_lm
import numpy as np

class TCML_TechnionIIT_lsqlm(OsipiBase):
    """
    TCML_TechnionIIT_lsqlm fitting algorithm by Angeleene Ang, Moti Freiman and Noam Korngut, TechnionIIT
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Angeleene Ang, Moti Freiman and Noam Korngut, TechnIIT"
    id_algorithm_type = "Bi-exponential fit with Levenberg-Marquardt algorithm"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    id_ref = "same github as https://doi.org/10.1007/978-3-031-16434-7_71, but not the main code from the paper"

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

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(TCML_TechnionIIT_lsqlm, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.fit_least_squares = fit_least_squares_lm
        self.fitS0=fitS0
        self.initialize(bounds, initial_guess, fitS0)

    def initialize(self, bounds, initial_guess, fitS0):
        self.use_bounds = {"f": False, "Dp": False, "D": False}

        if initial_guess is None:
            self.use_initial_guess = {"f": False, "Dp": False, "D": False}
        else:
            self.use_initial_guess = {"f": True, "Dp": True, "D": True}
        self.fitS0=fitS0

    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """

        bvalues=np.array(bvalues)
        initial_guess = [self.initial_guess["D"], self.initial_guess["Dp"], self.initial_guess["f"], self.initial_guess["S0"]]
        fit_results = self.fit_least_squares(bvalues, np.array(signals)[:,np.newaxis], initial_guess)

        def get_scalar(val):
            """Convert value to Python scalar, handling numpy arrays."""
            if isinstance(val, np.ndarray):
                return float(val.item())
            return float(val)

        results = {}
        if fit_results[0].size > 0:
            results["D"] = get_scalar(fit_results[0])
            results["f"] = get_scalar(fit_results[2])
            results["Dp"] = get_scalar(fit_results[1])
        else:
            results["D"] = 0
            results["f"] = 0
            results["Dp"] = 0

        results = self.D_and_Ds_swap(results)

        return results