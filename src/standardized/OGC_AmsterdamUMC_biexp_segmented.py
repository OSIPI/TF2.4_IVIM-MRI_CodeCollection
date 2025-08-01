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

        bvalues=np.array(bvalues)
        fit_results = self.OGC_algorithm(bvalues, signals, bounds=self.bounds, cutoff=self.thresholds, p0=self.initial_guess)

        results = {}
        results["D"] = fit_results[0]
        results["f"] = fit_results[1]
        results["Dp"] = fit_results[2]

        return results


    def ivim_fit_full_volume(self, signals, bvalues, njobs=4, **kwargs):
        """Perform the IVIM fit
        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.
        Returns:
            _type_: _description_
        """
        # normalize signals
        # Get index of b=0
        shape=np.shape(signals)

        b0_index = np.where(bvalues == 0)[0][0]
        # Mask of voxels where signal at b=0 >= 0.5
        valid_mask = signals[..., b0_index] >= 0.01
        # Select only valid voxels for fitting
        signals = signals[valid_mask]

        minimum_bvalue = np.min(bvalues) # We normalize the signal to the minimum bvalue. Should be 0 or very close to 0.
        b0_indices = np.where(bvalues == minimum_bvalue)[0]
        normalization_factor = np.mean(signals[..., b0_indices],axis=-1)
        signals = signals / np.repeat(normalization_factor[...,np.newaxis],np.shape(signals)[-1],-1)

        bvalues=np.array(bvalues)

        fit_results = np.array(fit_segmented_array(bvalues, signals, bounds=self.bounds, cutoff=self.thresholds, p0=self.initial_guess))

        D=np.zeros(shape[0:-1])
        D[valid_mask]=fit_results[0]
        f=np.zeros(shape[0:-1])
        f[valid_mask]=fit_results[1]
        Dp=np.zeros(shape[0:-1])
        Dp[valid_mask]=fit_results[2]
        results = {}
        results["D"] = D
        results["f"] = f
        results["Dp"] = Dp
        return results
