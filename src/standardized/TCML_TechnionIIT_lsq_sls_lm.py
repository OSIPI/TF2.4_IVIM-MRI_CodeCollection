from src.wrappers.OsipiBase import OsipiBase
from super_ivim_dc.source.Classsic_ivim_fit import IVIM_fit_sls_lm
import numpy as np

class TCML_TechnionIIT_lsq_sls_lm(OsipiBase):
    """
    TCML_TechnionIIT_lsqlm fitting algorithm by Angeleene Ang, Moti Freiman and Noam Korngut, TechnionIIT
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Angeleene Ang, Moti Freiman and Noam Korngut, TechnIIT"
    id_algorithm_type = "Bi-exponential, segmented as initaition, followed by Levenberg-Marquardt algorithm"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    id_ref = "same github as https://doi.org/10.1007/978-3-031-16434-7_71, but not the main code from the paper"

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

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(TCML_TechnionIIT_lsq_sls_lm, self).__init__(bvalues=bvalues, bounds=bounds)
        self.fit_least_squares = IVIM_fit_sls_lm
        self.fitS0=fitS0
        self.initialize(bounds, fitS0,thresholds)

    def initialize(self, bounds, fitS0,thresholds):
        if bounds is None:
            print(
                'warning, no bounds were defined, so default bounds are used of ([0.0003, 0.001, 0.009, 0],[0.008, 0.5,0.04, 3])')
            self.bounds = ([0.0003, 0.001, 0.009, 0], [0.008, 0.5, 0.04, 3])
        else:
            bounds = bounds
            self.bounds = bounds
        if thresholds is None:
            self.thresholds = 150
            print('warning, no thresholds were defined, so default bounds are used of  150')
        else:
            self.thresholds = thresholds
        self.fitS0=fitS0
        self.use_bounds = False
        self.use_initial_guess = False

    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        signals[signals<0]=0
        bvalues=np.array(bvalues)
        fit_results = self.fit_least_squares(np.array(signals)[:,np.newaxis],bvalues, self.bounds, min_bval_high=self.thresholds)

        results = {}
        if fit_results[0].size > 0:
            results["D"] = fit_results[0]
            results["f"] = fit_results[2]
            results["Dp"] = fit_results[1]
        else:
            results["D"] = 0
            results["f"] = 0
            results["Dp"] = 0

        return results