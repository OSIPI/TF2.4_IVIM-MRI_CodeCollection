from src.wrappers.OsipiBase import OsipiBase
from src.original.DT_IIITN.wls_ivim_fitting import wls_ivim_fit
import numpy as np


class DT_IIITN_WLS(OsipiBase):
    """
    Weighted Least Squares IVIM fitting using statsmodels Robust Linear Model.

    Segmented approach:
    1. Estimate D from high b-values using robust linear regression on log-signal
    2. Estimate D* from residuals at low b-values using robust linear regression

    Author: Devguru Tiwari, IIIT Nagpur

    Reference:
        Veraart, J. et al. (2013). "Weighted linear least squares estimation of
        diffusion MRI parameters: strengths, limitations, and pitfalls."
        NeuroImage, 81, 335-346.
        DOI: 10.1016/j.neuroimage.2013.05.028
    """

    # Algorithm identification
    id_author = "Devguru Tiwari, IIIT Nagpur"
    id_algorithm_type = "Weighted least squares segmented fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    id_ref = "https://doi.org/10.1016/j.neuroimage.2013.05.028"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0, 0]
    required_bounds = False
    required_bounds_optional = True
    required_initial_guess = False
    required_initial_guess_optional = True

    # Supported inputs
    supported_bounds = False
    supported_initial_guess = False
    supported_thresholds = True
    supported_dimensions = 1
    supported_priors = False

    def __init__(self, bvalues=None, thresholds=None,
                 bounds=None, initial_guess=None):
        """
        Initialize the WLS IVIM fitting algorithm.

        Args:
            bvalues (array-like, optional): b-values for the fitted signals.
            thresholds (array-like, optional): Threshold b-value for segmented
                fitting. The first value is used as the cutoff between high
                and low b-values. Default: 200 s/mm².
            bounds (dict, optional): Not used by this algorithm.
            initial_guess (dict, optional): Not used by this algorithm.
        """
        super(DT_IIITN_WLS, self).__init__(
            bvalues=bvalues, bounds=bounds,
            initial_guess=initial_guess, thresholds=thresholds
        )

    def ivim_fit(self, signals, bvalues, **kwargs):
        """Perform the IVIM fit using WLS.

        Args:
            signals (array-like): Signal intensities at each b-value.
            bvalues (array-like, optional): b-values for the signals.

        Returns:
            dict: Dictionary with keys "D", "f", "Dp".
        """
        bvalues = np.array(bvalues)

        # Use threshold as cutoff if available
        cutoff = 200
        if self.thresholds is not None and len(self.thresholds) > 0:
            cutoff = self.thresholds[0]

        D, f, Dp = wls_ivim_fit(bvalues, signals, cutoff=cutoff)

        results = {}
        results["D"] = D
        results["f"] = f
        results["Dp"] = Dp

        return results
