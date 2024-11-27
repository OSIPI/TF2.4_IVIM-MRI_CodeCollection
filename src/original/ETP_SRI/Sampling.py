import numpy as np
import emcee
import tqdm
import corner
from scipy.stats import rice, norm, uniform
from matplotlib import pyplot as plt

from utilities.data_simulation.GenerateData import GenerateData

class MCMC:
    """
    Performs sampling of exponential data
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
        # check this!
        # print(f'likelihood {norm.logpdf(self.data, loc=expected/self.data_scale, scale=self.data_scale)}')
        return np.sum(norm.logpdf(self.data, loc=expected, scale=self.data_scale), 1)
    
    # def biexp_loglike_gauss_full(self, f, D, D_star):
    #     expected = self.signal(f, D, D_star)
    #     print(f'expected {expected}')
    #     print(f'data {self.data}')
    #     return norm.logpdf(self.data, loc=expected, scale=self.data_scale)
    
    def biexp_loglike_rice(self, f, D, D_star):
        expected = self.signal(f, D, D_star)
        # print(f'expected {expected}')
        return np.sum(rice.logpdf(self.data, b=expected/self.data_scale, scale=self.data_scale), 1)
    
    def posterior(self, params):
        params = np.atleast_2d(params)
        total = self.bounds_prior(params)
        # print(f'bounds params {total}')
        neginf = np.isneginf(total)
        # print(f'neginf {neginf}')
        f = params[~neginf, 0]
        D = params[~neginf, 1]
        D_star = params[~neginf, 2]
        prior = self.prior(params[~neginf, :])
        # print(f'prior {prior}')
        likelihood = self.likelihood(f, D, D_star)
        # print(f'likelihood {likelihood}')
        total[~neginf] += prior + likelihood
        return total
        
    def sample(self, initial_pos):
        # f = initial_pos[0]
        # D = initial_pos[1]
        # D_star = initial_pos[2]
        # print(f'initial pos likelihood {self.biexp_loglike_gauss_full(f, D, D_star)}')
        print(f'initial pos likelihood {self.posterior(initial_pos)}')
        sampler = emcee.EnsembleSampler(self.nwalkers, 3, self.posterior, vectorize=True)
        pos = initial_pos + self.parameter_scale * np.random.randn(self.nwalkers, self.ndim)
        # print(f'pos {pos}')
        # print(f'nsteps {self.nsteps}')
        sampler.run_mcmc(pos, self.nsteps, progress=True)
        self.chain = sampler.get_chain(discard=self.burn_in, flat=True)
        self.means = np.mean(self.chain, 0)
        self.stds = np.std(self.chain, 0)
        print(f'final pos likelihood {self.posterior(self.means)}')
        # print(f'final pos likelihood {self.biexp_loglike_gauss_full(self.means[0], self.means[1], self.means[2])}')
        # print(f'chain {self.chain}')
        return self.means, self.stds
    
    def plot(self, truths=None, labels=('f', 'D', 'D*'), overplot=None):
        if truths is None:
            truths = self.means
        # print(f'chain size {self.chain.shape}')
        fig = corner.corner(self.chain, labels=labels, truths=truths)
        fig.suptitle("Sampling of the IVIM data", fontsize=16)
        if overplot is not None:
            corner.overplot_lines(fig, overplot, color='r')
        plt.show()
    