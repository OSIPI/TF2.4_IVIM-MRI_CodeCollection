import numpy as np
from src.wrappers.OsipiBase import OsipiBase
from src.original.PV_MUMC.two_step_IVIM_fit import fit_least_squares


class PV_MUMC_biexp(OsipiBase):
    """
    Bi-exponential fitting algorithm by Paulien Voorter, Maastricht University
    """
    
    # Some basic stuff that identifies the algorithm
    id_author = "Paulien Voorter MUMC"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"
    
    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,0] # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True

    # Supported inputs in the standardized class
    supported_bounds = True
    supported_initial_guess = False
    supported_thresholds = True
    supported_dimensions = 1
    supported_priors = False
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, weighting=None, stats=False):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(PV_MUMC_biexp, self).__init__(bvalues=bvalues, thresholds=thresholds, bounds=bounds, initial_guess=initial_guess)
        self.PV_algorithm = fit_least_squares

        self.use_bounds = {"f" : True, "D" : True, "Dp" : True, "S0" : True}
        self.use_initial_guess = {"f" : False, "D" : False, "Dp" : False, "S0" : False}
        

    def ivim_fit(self, signals, bvalues=None):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            dict: Fitted IVIM parameters f, Dp (D*), and D.
        """
        # --- bvalues resolution ---
        # Edge case: bvalues not passed here → fall back to the ones set at __init__ time
        if bvalues is None:
            if self.bvalues is None:
                raise ValueError(
                    "PV_MUMC_biexp: bvalues must be provided either at initialization or at fit time."
                )
            bvalues = self.bvalues
        else:
            bvalues = np.asarray(bvalues)

        # --- Bounds resolution ---
        # self.bounds is always a dict (OsipiBase force_default_settings=True).
        # The underlying fit_least_squares expects: ([S0min, Dmin, fmin, Dpmin], [S0max, Dmax, fmax, Dpmax])
        if isinstance(self.bounds, dict):
            bounds = (
                [self.bounds["S0"][0], self.bounds["D"][0], self.bounds["f"][0], self.bounds["Dp"][0]],
                [self.bounds["S0"][1], self.bounds["D"][1], self.bounds["f"][1], self.bounds["Dp"][1]],
            )
        else:
            # Fallback: already in list/tuple form (legacy)
            bounds = self.bounds

        if self.thresholds is None:
            self.thresholds = 200

        # Default fallback parameters (D, f, Dp) used if the optimizer fails
        DEFAULT_PARAMS = [0.001, 0.1, 0.01]

        try:
            fit_results = self.PV_algorithm(bvalues, signals, bounds=bounds, cutoff=self.thresholds)
        except RuntimeError as e:
            # curve_fit raises RuntimeError both for max-evaluations exceeded and other failures
            print(f"PV_MUMC_biexp: optimizer failed ({e}). Returning default parameters.")
            fit_results = DEFAULT_PARAMS
        except Exception as e:
            # Catch any other unexpected error (e.g. all-zero signal, NaNs in input)
            print(f"PV_MUMC_biexp: unexpected error during fit ({type(e).__name__}: {e}). Returning default parameters.")
            fit_results = DEFAULT_PARAMS

        results = {}
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]
        results["D"] = fit_results[0]

        return results
