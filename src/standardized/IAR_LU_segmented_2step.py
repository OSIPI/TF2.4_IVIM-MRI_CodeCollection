import numpy as np
from dipy.core.gradients import gradient_table
from src.wrappers.OsipiBase import OsipiBase
from src.original.IAR_LundUniversity.ivim_fit_method_segmented_2step import IvimModelSegmented2Step


class IAR_LU_segmented_2step(OsipiBase):
    """
    Bi-exponential fitting algorithm by Ivan A. Rashid, Lund University
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Ivan A. Rashid, LU"
    id_algorithm_type = "Segmented bi-exponential fit"
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
    supported_initial_guess = True
    supported_thresholds = True
    supported_dimensions = 1
    supported_priors = False
    
    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(IAR_LU_segmented_2step, self).__init__(bvalues, thresholds, bounds, initial_guess)
        if bounds is None:
            self.use_bounds = {"f": False, "Dp": False, "D": False, "S0": False}
        else:
            self.use_bounds = {"f": True, "Dp": True, "D": True, "S0": True}

        if initial_guess is None:
            self.use_initial_guess = {"f": False, "Dp": False, "D": False, "S0": False}
        else:
            self.use_initial_guess = {"f": True, "Dp": True, "D": True, "S0": True}

        # Check the inputs

        
        # Initialize the algorithm
        if self.bvalues is not None:
            bvec = np.zeros((self.bvalues.size, 3))
            bvec[:,2] = 1
            gtab = gradient_table(self.bvalues, bvec, b0_threshold=0)

            # Convert dict bounds/initial_guess to list-of-lists as expected by IvimModelSegmented2Step
            bounds_list = [[self.bounds["S0"][0], self.bounds["f"][0], self.bounds["Dp"][0], self.bounds["D"][0]],
                           [self.bounds["S0"][1], self.bounds["f"][1], self.bounds["Dp"][1], self.bounds["D"][1]]]
            initial_guess_list = [self.initial_guess["S0"], self.initial_guess["f"], self.initial_guess["Dp"], self.initial_guess["D"]]

            self.IAR_algorithm = IvimModelSegmented2Step(gtab, bounds=bounds_list, initial_guess=initial_guess_list, b_threshold=self.thresholds)
        else:
            self.IAR_algorithm = None
        
    
    def ivim_fit(self, signals, bvalues, thresholds=None, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used.

        Returns:
            dict: Fitted IVIM parameters f, Dp (D*), and D.
        """
        # --- bvalues resolution ---
        if bvalues is None:
            if self.bvalues is None:
                raise ValueError(
                    "IAR_LU_segmented_2step: bvalues must be provided either at initialization or at fit time."
                )
            bvalues = self.bvalues
        else:
            bvalues = np.asarray(bvalues)

        # Adapt the bounds to the format needed for the algorithm
        bounds = [[self.bounds["S0"][0], self.bounds["f"][0], self.bounds["Dp"][0], self.bounds["D"][0]],
                  [self.bounds["S0"][1], self.bounds["f"][1], self.bounds["Dp"][1], self.bounds["D"][1]]]

        # Adapt the initial guess to the format needed for the algorithm
        initial_guess = [self.initial_guess["S0"], self.initial_guess["f"], self.initial_guess["Dp"], self.initial_guess["D"]]

        # Guard: reinitialise if the algorithm is not yet built, OR if bvalues have changed
        # (calling with different bvalues than __init__ must rebuild the gradient table)
        current_bvals = None if self.IAR_algorithm is None else self.IAR_algorithm.bvals
        bvalues_changed = (current_bvals is not None) and not np.array_equal(current_bvals, bvalues)

        if self.IAR_algorithm is None or bvalues_changed:
            bvec = np.zeros((bvalues.size, 3))
            bvec[:,2] = 1
            gtab = gradient_table(bvalues, bvec, b0_threshold=0)

            if self.thresholds is None:
                self.thresholds = 200

            self.IAR_algorithm = IvimModelSegmented2Step(
                gtab, bounds=bounds, initial_guess=initial_guess, b_threshold=self.thresholds
            )

        try:
            fit_results = self.IAR_algorithm.fit(signals)
        except Exception as e:
            print(f"IAR_LU_segmented_2step: fit failed ({type(e).__name__}: {e}). Returning default parameters.")
            results = {"f": self.initial_guess["f"], "Dp": self.initial_guess["Dp"], "D": self.initial_guess["D"]}
            return results

        results = {}
        results["f"] = fit_results.model_params[1]
        results["Dp"] = fit_results.model_params[2]
        results["D"] = fit_results.model_params[3]
        # Ensure D < Dp (swap if the optimizer returned them in wrong order)
        results = self.D_and_Ds_swap(results)

        return results