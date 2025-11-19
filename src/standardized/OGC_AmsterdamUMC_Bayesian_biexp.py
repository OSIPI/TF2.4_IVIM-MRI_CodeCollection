from src.wrappers.OsipiBase import OsipiBase
from src.original.OGC_AmsterdamUMC.LSQ_fitting import flat_neg_log_prior, fit_bayesian, empirical_neg_log_prior, fit_segmented, fit_bayesian_array, fit_segmented_array
import numpy as np

class OGC_AmsterdamUMC_Bayesian_biexp(OsipiBase):
    """
    Bayesian Bi-exponential fitting algorithm by Oliver Gurney-Champion, Amsterdam UMC
    """

    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements

    # Some basic stuff that identifies the algorithm
    id_author = "Oliver Gurney Champion, Amsterdam UMC"
    id_algorithm_type = "Bi-exponential fit"
    id_return_parameters = "f, D*, D, S0"
    id_units = "seconds per milli metre squared or milliseconds per micro metre squared"

    # Algorithm requirements
    required_bvalues = 4
    required_thresholds = [0,
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
    supported_priors = True

    def __init__(self, bvalues=None, thresholds=None, bounds=None, initial_guess=None, fitS0=True, prior_in=None):

        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.

            Args:
                 datain (Array): is a 2D array with values of D, f, D* (and S
                 ) that will form the prior.
                 thresholds (Bolean, optional): a bolean indicating what threshold is used
                 prior_in (array, optional): 2D array of D, f, D* and (optionally) S0 values which form the prior

        """
        super(OGC_AmsterdamUMC_Bayesian_biexp, self).__init__(bvalues=bvalues, bounds=bounds, thresholds=thresholds, initial_guess=initial_guess) #, fitS0, prior_in)
        self.OGC_algorithm = fit_bayesian
        self.OGC_algorithm_array = fit_bayesian_array
        self.initialize(bounds, initial_guess, fitS0, prior_in, thresholds)
        self.fit_segmented=fit_segmented

    def initialize(self, bounds=None, initial_guess=None, fitS0=True, prior_in=None, thresholds=None):
        self.use_initial_guess = {"f" : True, "D" : True, "Dp" : True, "S0" : True}
        self.use_bounds = {"f" : True, "D" : True, "Dp" : True, "S0" : True}
        self.thresholds = thresholds

        if prior_in is None:
            print('using a flat prior between bounds')
            self.neg_log_prior=flat_neg_log_prior([self.bounds["D"][0],self.bounds["D"][1]],[self.bounds["f"][0],self.bounds["f"][1]],[self.bounds["Dp"][0],self.bounds["Dp"][1]],[self.bounds["S0"][0],self.bounds["S0"][1]])
        else:
            print('warning, bounds are not used, as a prior is used instead')
            if len(prior_in) == 4:
                self.neg_log_prior = empirical_neg_log_prior(prior_in[0], prior_in[1], prior_in[2],prior_in[3])
            else:
                self.neg_log_prior = empirical_neg_log_prior(prior_in[0], prior_in[1], prior_in[2])
        self.fitS0=fitS0

    def ivim_fit(self, signals, bvalues, initial_guess=None, **kwargs):
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

        epsilon = 0.000001
        fit_results = fit_segmented(bvalues, signals, bounds=bounds, cutoff=self.thresholds, p0=initial_guess)
        fit_results=np.array(fit_results+(1,))
        for i in range(4):
            if fit_results[i] < bounds[0][i] : fit_results[0] = bounds[0][i]+epsilon
            if fit_results[i] > bounds[1][i] : fit_results[0] = bounds[1][i]-epsilon
        fit_results = self.OGC_algorithm(bvalues, signals, self.neg_log_prior, x0=fit_results, fitS0=self.fitS0, bounds=bounds)

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
        bounds = ([self.bounds["D"][0], self.bounds["f"][0], self.bounds["Dp"][0], self.bounds["S0"][0]],
                  [self.bounds["D"][1], self.bounds["f"][1], self.bounds["Dp"][1], self.bounds["S0"][1]])

        initial_guess = [self.initial_guess["D"], self.initial_guess["f"], self.initial_guess["Dp"], self.initial_guess["S0"]]

        # normalize signals
        # Get index of b=0
        shape=np.shape(signals)

        b0_index = np.where(bvalues == 0)[0][0]
        # Mask of voxels where signal at b=0 >= 0.5
        valid_mask = signals[..., b0_index] >= 0
        # Select only valid voxels for fitting
        signals = signals[valid_mask]

        minimum_bvalue = np.min(bvalues) # We normalize the signal to the minimum bvalue. Should be 0 or very close to 0.
        b0_indices = np.where(bvalues == minimum_bvalue)[0]
        normalization_factor = np.mean(signals[..., b0_indices],axis=-1)
        signals = signals / np.repeat(normalization_factor[...,np.newaxis],np.shape(signals)[-1],-1)

        bvalues=np.array(bvalues)

        epsilon = 0.000001
        fit_results = np.array(fit_segmented_array(bvalues, signals, bounds=bounds, cutoff=self.thresholds, p0=initial_guess))
        #fit_results=np.array(fit_results+(1,))
        # Loop over parameters (rows)

        for i in range(4):
            if i == 3:
                fit_results[i] = np.random.normal(1,0.2,np.shape(fit_results[i]))
            else:
                below = fit_results[i] < bounds[0][i]
                above = fit_results[i] > bounds[1][i]

                fit_results[i, below] = bounds[0][i] + epsilon
                fit_results[i, above] = bounds[1][i] - epsilon
        self.jobs=njobs
        fit_results = self.OGC_algorithm_array(bvalues, signals,fit_results, self)

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


    def reshape_to_voxelwise(self, data):
        """
        reshapes multi-D input (spatial dims, bvvalue) data to 2D voxel-wise array
        Args:
            data (array): mulit-D array (data x b-values)
        Returns:
            out (array): 2D array (voxel x b-value)
        """
        B = data.shape[-1]
        voxels = int(np.prod(data.shape[:-1]))  # e.g., X*Y*Z
        return data.reshape(voxels, B)
