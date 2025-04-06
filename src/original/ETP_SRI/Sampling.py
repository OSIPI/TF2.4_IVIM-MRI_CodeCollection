import numpy as np
import emcee
import corner
from scipy.stats import rice, norm, uniform
from matplotlib import pyplot as plt

from utilities.data_simulation.GenerateData import GenerateData

class MCMC:
    """
    Performs sampling of exponential data using MCMC
    """
    def __init__(self,
                 data,
                 b,
                 data_scale=1e-5,
                 parameter_scale=(1e-7, 1e-11, 1e-9),
                 bounds=((0, 1), (0, 1), (0, 1)),
                 priors=None,
                 gaussian_noise=False,
                 nwalkers=16,
                 nsteps=10000,
                 burn_in=2000,
                 progress=True):
        """
        Parameters
        ----------
        data : array_like
            The data to be fitted
        b : array_like
            The b-values for the data
        data_scale : float, optional
            The scale of the data, by default 1e-5
        parameter_scale : tuple, optional
            The scale of the parameters, by default (1e-7, 1e-11, 1e-9)
        bounds : tuple, optional
            The bounds for the parameters, by default ((0, 1), (0, 1), (0, 1))
        priors : tuple, optional
            The priors for the parameters, by default None
        gaussian_noise : bool, optional
            Whether the noise is gaussian, by default False
        nwalkers : int, optional
            The number of walkers, by default 16
        nsteps : int, optional
            The number of steps, by default 10000
        burn_in : int, optional
            The burn in, by default 2000
        progress : bool, optional
            Whether to show progress, by default True

        """
        self.data = np.atleast_2d(np.asarray(data))
        self.b = np.atleast_2d(np.asarray(b)).T
        self.data_scale = np.asarray(data_scale)
        self.parameter_scale = np.asarray(parameter_scale)
        self.bounds = np.asarray(bounds)
        self.priors = np.asarray(priors)
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burn_in = burn_in
        self.progress = progress
        if priors is None:
            self.prior = self.zero_prior
        else:
            self.prior = self.loglike_gauss_prior
        if gaussian_noise:
            self.likelihood = self.biexp_loglike_gauss
        else:
            self.likelihood = self.biexp_loglike_rice
        self.ndim = 3
        self.chain = None
        self.means = None
        self.stds = None

    def accepted_dimensions(self):
        return (1, 1)
    
    def bounds_prior(self, params):
        return np.sum(uniform.logpdf(params, loc=self.bounds[:, 0], scale=self.bounds[:, 1] - self.bounds[:, 0]), 1)
    
    def loglike_gauss_prior(self, params):
        return np.sum(norm.logpdf(params, loc=self.priors[:, 0], scale=self.priors[:, 1]), 1)
    
    def zero_prior(self, params):
        return 0
    
    def signal(self, f, D, D_star):
        return (f * np.exp(-self.b * D_star) + (1 - f) * np.exp(-self.b * D)).T
    
    def biexp_loglike_gauss(self, f, D, D_star):
        expected = self.signal(f, D, D_star)
        return np.sum(norm.logpdf(self.data, loc=expected, scale=self.data_scale), 1)
    
    def biexp_loglike_rice(self, f, D, D_star):
        expected = self.signal(f, D, D_star)
        return np.sum(rice.logpdf(self.data, b=expected/self.data_scale, scale=self.data_scale), 1)
    
    def posterior(self, params):
        params = np.atleast_2d(params)
        total = self.bounds_prior(params)
        neginf = np.isneginf(total)
        f = params[~neginf, 0]
        D = params[~neginf, 1]
        D_star = params[~neginf, 2]
        prior = self.prior(params[~neginf, :])
        likelihood = self.likelihood(f, D, D_star)
        total[~neginf] += prior + likelihood
        return total
        
    def sample(self, initial_pos):
        # print(f'initial pos likelihood {self.posterior(initial_pos)}')
        sampler = emcee.EnsembleSampler(self.nwalkers, 3, self.posterior, vectorize=True)
        pos = initial_pos + self.parameter_scale * np.random.randn(self.nwalkers, self.ndim)
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.chain = sampler.get_chain(discard=self.burn_in, flat=True)
        self.means = np.mean(self.chain, 0)
        self.stds = np.std(self.chain, 0)
        # print(f'final pos likelihood {self.posterior(self.means)}')
        return self.means, self.stds
    
    def plot(self, truths=None, labels=('f', 'D', 'D*'), overplot=None, title='Sampling of the IVIM data'): #, show=True, save_path=None, close=False):
        if truths is None:
            truths = self.means
        fig = corner.corner(self.chain, labels=labels, truths=truths)
        fig.suptitle(title, fontsize=16)
        if overplot is not None:
            corner.overplot_lines(fig, overplot, color='r')
        return fig
        # if save_path is not None:
        #     fig.savefig(save_path)
        # if show:
        #     fig.show()
        # if close:
        #     plt.close(fig)
    