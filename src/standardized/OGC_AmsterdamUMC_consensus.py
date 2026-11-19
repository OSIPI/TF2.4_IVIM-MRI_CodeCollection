from src.wrappers.OsipiBase import OsipiBase
import warnings
import numpy as np
from scipy.optimize import curve_fit

class OGC_AmsterdamUMC_consensus(OsipiBase):
    """
    Consensus-based segmented bi-exponential fitting algorithm by Oliver Gurney-Champion, Amsterdam UMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC13175230/)
    """


    # Some basic stuff that identifies the algorithm
    id_author = "Oliver Gurney Champion, Amsterdam UMC"
    id_algorithm_type = "consensus fit"
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
    supported_initial_guess = False
    supported_thresholds = True
    supported_dimensions = 1
    supported_priors = False

    def __init__(self, bvalues=None, thresholds=200, bounds=None, initial_guess=None):
        """
            Everything this algorithm requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.

            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(OGC_AmsterdamUMC_consensus, self).__init__(bvalues, thresholds, bounds, initial_guess)
        self.OGC_algorithm = self.algorithm
        self.initialize(thresholds)

    def algorithm(self, data):

        data=data/np.mean(data[self.bvalues==0])
        high_b = self.bvalues[self.bvalues >= self.thresholds]
        high_data = data[self.bvalues >= self.thresholds]
        # correct the bounds. Note that S0 bounds determine the max and min of f
        results = {}

        intercept, Dt = self.iWLLS2(high_b,high_data)
        Fp = 1 - intercept

        if Fp < self.bounds["f"][0] : Fp = self.bounds["f"][0]
        if Fp > self.bounds["f"][1] : Fp = self.bounds["f"][1]
        if Dt < self.bounds["D"][0] : Dt = self.bounds["D"][0]
        if Dt > self.bounds["D"][1] : Dt = self.bounds["D"][1]
        results["D"] = Dt
        results["f"] = Fp

        # remove the diffusion part to only keep the pseudo-diffusion
        p0 = max(min(10*Dt,self.bounds["Dp"][1]),self.bounds["Dp"][0])
        bounds = (self.bounds["Dp"][0], self.bounds["Dp"][1])
        # fit for D*
        params, _ = curve_fit(lambda b, Dp: Fp * np.exp(-b * Dp) + (1-Fp) * np.exp(-b * Dt), self.bvalues, data, p0=p0, bounds=bounds)
        results["Dp"] = params[0]
        return results

    def iWLLS2(self, bvals, data):
        y=np.log(data)
        bvals=np.array(bvals)
        x=np.vstack([-bvals, np.ones(len(bvals))]).T

        D_lls, S0 = np.linalg.lstsq(x,y)[0]
        S_est = np.exp(S0) * np.exp(-bvals * D_lls)
        eps=1
        i=0
        while eps>1e-7 and i<25:
            W=np.diag(S_est ** 2)
            D_wlls, wS0 = np.linalg.inv(x.T @ W @ x) @ x.T @ W @ y
            # XtW= x.T @ W
            # D_wlls, wS0 = np.linalg.lstsq(XtW @ S_est ,XtW @ y)[0]
            S_est = np.exp(wS0) * np.exp(-bvals * D_wlls)
            eps=abs(D_lls-D_wlls)
            D_lls=D_wlls
            i=i+1
        return np.exp(wS0), D_wlls

    def initialize(self, thresholds):
        self.use_initial_guess = {"f" : True, "D" : False, "Dp" : True, "S0" : False}
        self.use_bounds = {"f" : True, "D" : False, "Dp" : True, "S0" : False}

        if self.thresholds is None:
            self.thresholds = 200
            warnings.warn('No thresholds were defined, so default threshold of 200 was used', UserWarning, stacklevel=2)
        else:
            self.thresholds = thresholds

    def ivim_fit(self, signals, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)

        Returns:
            _type_: _description_
        """

        results = self.OGC_algorithm(signals)


        return results