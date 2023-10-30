from src.wrappers.OsipiBase import OsipiBase
from src.original.OGC_AmsterdamUMC.LSQ_fitting import flat_neg_log_prior, fit_bayesian, empirical_neg_log_prior, fit_segmented
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
                           0]  # Interval from "at least" to "at most", in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True  # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = True
    accepted_dimensions = 1  # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    accepts_priors = True

    def __init__(self, bvalues=None, thresholds=None, bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.2, 1.3]), initial_guess=None, fitS0=True, prior_in=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.

            Args:
                 datain is a 2D array with values of D, f, D* (and S0) that will form the prior.
        """
        super(OGC_AmsterdamUMC_Bayesian_biexp, self).__init__(bvalues, bounds, initial_guess) #, fitS0, prior_in)
        self.OGC_algorithm = fit_bayesian
        self.initialize(bounds, initial_guess, fitS0, prior_in)

    def initialize(self, bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.2, 1.3]), initial_guess=None, fitS0=True, prior_in=None):
        if bounds is None:
            self.bounds=([0, 0, 0.005, 0.7],[0.005, 1, 0.2, 1.3])
        else:
            self.bounds=bounds
        if prior_in is None:
            self.neg_log_prior=flat_neg_log_prior([self.bounds[0][0],self.bounds[1][0]],[self.bounds[0][1],self.bounds[1][1]],[self.bounds[0][2],self.bounds[1][2]],[self.bounds[0][3],self.bounds[1][3]])
        else:
            if len(prior_in) is 4:
                self.neg_log_prior = empirical_neg_log_prior(prior_in[0], prior_in[1], prior_in[2],prior_in[3])
            else:
                self.neg_log_prior = empirical_neg_log_prior(prior_in[0], prior_in[1], prior_in[2])
        if initial_guess is None:
            self.initial_guess = [0.001, 0.5, 0.1, 1]
        else:
            self.initial_guess = initial_guess
        self.fitS0=fitS0

    def ivim_fit(self, signals, bvalues, initial_guess=None, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.

        Returns:
            _type_: _description_
        """
        if initial_guess is not None and len(initial_guess) == 4:
            self.initial_guess = initial_guess
        bvalues=np.array(bvalues)
        fit_results = fit_segmented(bvalues, signals, bounds=self.bounds, cutoff=150, p0=self.initial_guess)
        fit_results=fit_results+(1,)
        fit_results = self.OGC_algorithm(bvalues, signals, self.neg_log_prior, x0=fit_results, fitS0=self.fitS0)

        D = fit_results[0]
        f = fit_results[1]
        Dstar = fit_results[2]

        return f, Dstar, D