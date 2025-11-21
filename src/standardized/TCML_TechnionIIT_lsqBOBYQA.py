from src.wrappers.OsipiBase import OsipiBase
from super_ivim_dc.source.Classsic_ivim_fit import fit_least_squares_BOBYQA
import numpy as np

class TCML_TechnionIIT_lsqBOBYQA(OsipiBase):
    """
    TCML_TechnionIIT_lsqBOBYQA fitting algorithm by Moti Freiman and Noam Korngut, TechnionIIT
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Moti Freiman and Noam Korngut, TechnIIT"
    id_algorithm_type = "Bi-exponential fit, BOBYQO"
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

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(TCML_TechnionIIT_lsqBOBYQA, self).__init__(bvalues=bvalues, bounds=bounds, initial_guess=initial_guess)
        self.fit_least_squares = fit_least_squares_BOBYQA
        self.fitS0=fitS0
        self.initialize(bounds, initial_guess, fitS0)

    def initialize(self, bounds, initial_guess, fitS0):
        if bounds is None:
            self.use_bounds = {"f": False, "Dp": False, "D": False}
        else:
            self.use_bounds = {"f": True, "Dp": True, "D": True}

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
        bounds = ([self.bounds["D"][0], self.bounds["Dp"][0], self.bounds["f"][0], self.bounds["S0"][0]],
                       [self.bounds["D"][1], self.bounds["Dp"][1], self.bounds["f"][1], self.bounds["S0"][1]])
        initial_guess = [self.initial_guess["D"], self.initial_guess["Dp"], self.initial_guess["f"], self.initial_guess["S0"]]

        fit_results = self.fit_least_squares(bvalues, np.array(signals)[:,np.newaxis], bounds, initial_guess.copy())

        results = {}
        if fit_results[0].size == 0:
            results["D"] = initial_guess[0]
            results["f"] = initial_guess[2]
            results["Dp"] = initial_guess[1]
        else:
            results["D"] = fit_results[0]
            results["f"] = fit_results[2]
            results["Dp"] = fit_results[1]
        results = self.D_and_Ds_swap(results)

        return results