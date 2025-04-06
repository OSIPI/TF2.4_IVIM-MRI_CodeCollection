import numpy as np
# import emcee
from src.wrappers.OsipiBase import OsipiBase
from src.original.ETP_SRI.Sampling import MCMC


class ETP_SRI_MCMC(OsipiBase):
    """WIP
    Implementation and execution of the submitted algorithm
    """
    
    # I'm thinking that we define default attributes for each submission like this
    # And in __init__, we can call the OsipiBase control functions to check whether
    # the user inputs fulfil the requirements
    
    # Some basic stuff that identifies the algorithm
    id_author = "Eric T. Peterson, SRI"
    id_algorithm_type = "Markov Chain Monte Carlo"
    id_return_parameters = "Mean, Standard deviation"
    id_units = "seconds per milli metre squared"
    
    # Algorithm requirements
    required_bvalues = 3
    required_thresholds = [0,1] # Interval from 1 to 1, in case submissions allow a custom number of thresholds
    required_bounds = False
    required_bounds_optional = True # Bounds may not be required but are optional
    required_initial_guess = False
    required_initial_guess_optional = False
    accepted_dimensions = 1
    # Not sure how to define this for the number of accepted dimensions. Perhaps like the thresholds, at least and at most?
    
    def __init__(self,
                 bvalues=None,
                 data_scale=1e-2,
                 parameter_scale=(1e-7, 1e-11, 1e-9),
                 bounds=((0, 1), (0, 1), (0, 1)),
                 priors=None,
                 gaussian_noise=False,
                 nwalkers=16,
                 nsteps=10000,
                 burn_in=2000,
                 progress=True,
                 initial_guess=None):
        """
            Everything this method requires should be implemented here.
            Number of segmentation thresholds, bounds, etc.
            
            Our OsipiBase object could contain functions that compare the inputs with
            the requirements.
        """
        super(ETP_SRI_MCMC, self).__init__(bvalues)
        self.result_keys = ['means', 'stds']
        
        # Could be a good idea to have all the submission-specfic variable be 
        # defined with initials?
        self.bvalues = bvalues
        self.data_scale = data_scale
        self.parameter_scale = parameter_scale
        self.bounds = bounds
        self.priors = priors
        self.gaussian_noise = gaussian_noise
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burn_in = burn_in
        self.progress = progress
        self.initial_guess = initial_guess

        self.MCMC = None
        
        # Check the inputs
        
    
    def ivim_fit(self, signals, initial_guess=None, bvalues=None, **kwargs):
        """Perform the IVIM fit

        Args:
            signals (array-like)
            bvalues (array-like, optional): b-values for the signals. If None, self.bvalues will be used. Default is None.
            linear_fit_option (bool, optional): This fit has an option to only run a linear fit. Defaults to False.

        Returns:
            _type_: _description_
        """
        if bvalues is None:
            bvalues = self.bvalues

        if initial_guess is not None:
            self.initial_guess = initial_guess
        
        self.__dict__.update(kwargs)
        
        self.MCMC = MCMC(signals, bvalues, self.data_scale, self.parameter_scale, self.bounds, self.priors, self.gaussian_noise, self.nwalkers, self.nsteps, self.burn_in, self.progress)

        means, stds = self.MCMC.sample(self.initial_guess)

        return {'means': means, 'stds': stds}
    
    def plot(self, **kwargs):
        """Plot the results of the MCMC fit

        Args:
            truths (array-like, optional): True values. Defaults to None.
            labels (tuple, optional): Labels for the parameters. Defaults to ('f', 'D', 'D*').
            overplot (array-like, optional): Overplot the true values. Defaults to None.
            title (str, optional): Title of the plot. Defaults to 'MCMC Fit'.
            show (bool, optional): Show the plot. Defaults to True.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        if self.MCMC is None:
            raise ValueError('No MCMC fit has been performed. Fit the data first.')
        return self.MCMC.plot(**kwargs)
    
