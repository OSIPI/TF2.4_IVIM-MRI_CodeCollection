from src.wrappers.OsipiBase import OsipiBase
from super_ivim_dc.source.Classsic_ivim_fit import IVIM_fit_sls_trf
import numpy as np

class TCML_TechnionIIT_lsq_sls_trf(OsipiBase):
    """
    TCML_TechnionIIT_lsqlm fitting algorithm by Angeleene Ang, Moti Freiman and Noam Korngut, TechnionIIT
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Angeleene Ang, Moti Freiman and Noam Korngut, TechnIIT"
    id_algorithm_type = "Bi-exponential fit, SLS fit followed by Trust Region Reflective algorithm"
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

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(TCML_TechnionIIT_lsq_sls_trf, self).__init__(bvalues=bvalues, bounds=bounds)
        self.fit_least_squares = IVIM_fit_sls_trf
        self.fitS0=fitS0
        self.initialize(bounds, fitS0, thresholds)

    def initialize(self, bounds, fitS0, thresholds):
        if bounds is None:
            print('warning, no bounds were defined, so default bounds are used of ([0.0003, 0.001, 0.009, 0],[0.008, 1.0,0.04, 3])')
            self.bounds = ([0.0003, 0.001, 0.009, 0],[0.008, 1.0,0.04, 3])
        else:
            bounds=bounds
            self.bounds = bounds
        if thresholds is None:
            self.thresholds = 200
            print('warning, no thresholds were defined, so default bounds are used of  200')
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
        bounds=np.array(self.bounds)
        bounds=[bounds[0][[0, 2, 1, 3]], bounds[1][[0, 2, 1, 3]]]
        fit_results = self.fit_least_squares(np.array(signals)[:,np.newaxis],bvalues, bounds,min_bval_high=self.thresholds)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[2]
        results["Dp"] = fit_results[1]

        return results