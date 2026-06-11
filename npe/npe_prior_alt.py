"""
npe_prior_alt.py
================
Alternative *tissue-informed* prior for the prior-sensitivity ablation
(Supplementary Figure S5), the one axis the architecture (S1) and acquisition
(S2) ablations leave uncontrolled.

The primary model (setB) uses the log-uniform-style ``BoxUniform`` prior in
``npe_prior.py``: with ``--log-dstar`` the D* axis is uniform on log10(D*) over
[log10(3e-3), log10(0.15)] — equal prior mass per decade, maximally
non-committal about where D* sits.

This module changes ONLY the D* marginal to a substantively different *shape*:
a **tissue-informed truncated Normal on log10(D*)**, peaked at a physiological
pseudo-diffusion value and down-weighting the decade extremes. D and f are left
as the identical uniforms, so the ablation isolates the effect of the D* prior —
the parameter whose below-floor overconfidence is the manuscript's thesis.

    D            ~ Uniform(0.2e-3, 3.0e-3)                      [absolute mm^2/s]
    log10(D*)    ~ TruncatedNormal(mu, sigma, low, high)        [prior space]
    f            ~ Uniform(0.0, 0.5)

with, by default,
    mu    = log10(0.03)  = -1.5229   (D* = 30e-3 mm^2/s, a physiological D*)
    sigma = 0.30 dex                 (1 sigma ~ a factor of 2: 15-60e-3;
                                      2 sigma ~ 7.5-120e-3, covering the
                                      efficiency grid's D* range 10-120e-3)
    [low, high] = [log10(3e-3), log10(0.15)] = [-2.5229, -0.8239]
                  (the SAME box as setB, so every draw stays in range for the
                  simulator and the audit grid).

The truncation to the setB box matters twice: it keeps the simulator/grid in
range, and it makes the prior support identical to setB's in log10(D*) space, so
``run_e_efficiency``'s log_dstar auto-detection (``support lower bound[1] < 0``)
still fires. This prior is ONLY meaningful with ``--log-dstar`` (its D* component
lives in log10 space); the caller is responsible for using log space.

Breadth note (for honest interpretation): a sharply informative prior would let
the Bayesian posterior legitimately sit below the prior-free CRLB, confounding
"overconfidence" with ordinary prior shrinkage. sigma = 0.30 dex is deliberately
broad (the log-uniform's own effective std over the box is ~0.49 dex), so
"below-floor" keeps roughly the same meaning across the two priors.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions import Distribution, Normal, Uniform
from sbi.utils import process_prior

# Re-export the box bounds so this module stays in lock-step with npe_prior.py.
from npe_prior import PARAM_NAMES, N_PARAMS, PRIOR_LOW, PRIOR_HIGH

# Default tissue-informed hyperparameters for the log10(D*) marginal.
DSTAR_PHYS = 0.03                       # physiological D* = 30e-3 mm^2/s
ALT_DSTAR_LOG_MU = float(np.log10(DSTAR_PHYS))      # -1.5229
ALT_DSTAR_LOG_SIGMA = 0.30                          # dex
# Truncate to the SAME log10(D*) box as setB.
ALT_DSTAR_LOG_LOW = float(np.log10(PRIOR_LOW[1]))   # log10(3e-3)  = -2.5229
ALT_DSTAR_LOG_HIGH = float(np.log10(PRIOR_HIGH[1]))  # log10(0.15) = -0.8239


class TruncatedNormal(Distribution):
    """Univariate truncated Normal with analytic sample()/log_prob().

    Built on ``torch.distributions.Normal`` via its CDF (for the normaliser and
    inverse-CDF sampling) and ICDF. ``event_shape`` is scalar so it slots into
    sbi's ``MultipleIndependent`` exactly like a 1-D ``Uniform``.
    """

    arg_constraints: dict = {}
    has_rsample = False

    def __init__(self, loc, scale, low, high, validate_args: bool = False):
        self.loc = torch.as_tensor(loc, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self._base = Normal(self.loc, self.scale)
        self._cdf_low = self._base.cdf(self.low)
        self._cdf_high = self._base.cdf(self.high)
        self._Z = (self._cdf_high - self._cdf_low).clamp_min(1e-12)
        super().__init__(batch_shape=self.loc.shape, event_shape=torch.Size(),
                         validate_args=validate_args)

    @property
    def support(self):
        return constraints.interval(self.low, self.high)

    @property
    def mean(self):
        # E[X] of a truncated normal; only needed for diagnostics.
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        phi = lambda z: torch.exp(Normal(0.0, 1.0).log_prob(z))
        return self.loc + self.scale * (phi(a) - phi(b)) / self._Z

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        # Inverse-CDF sampling within the truncation window.
        p = (self._cdf_low + u * self._Z).clamp(1e-7, 1.0 - 1e-7)
        return self._base.icdf(p)

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.loc.dtype)
        lp = self._base.log_prob(value) - torch.log(self._Z)
        outside = (value < self.low) | (value > self.high)
        return torch.where(outside, torch.full_like(lp, float("-inf")), lp)


def _component_priors(device: str = "cpu",
                      dstar_log_mu: float = ALT_DSTAR_LOG_MU,
                      dstar_log_sigma: float = ALT_DSTAR_LOG_SIGMA):
    """The three independent 1-D priors, D and f IDENTICAL to the setB box.

    sbi requires each component to carry ``batch_shape=[1]`` (not scalar), so
    every parameter is wrapped in a 1-element tensor.
    """
    def t(x):
        return torch.tensor([x], dtype=torch.float32, device=device)

    d = Uniform(t(PRIOR_LOW[0]), t(PRIOR_HIGH[0]))
    log_dstar = TruncatedNormal(loc=t(dstar_log_mu), scale=t(dstar_log_sigma),
                                low=t(ALT_DSTAR_LOG_LOW), high=t(ALT_DSTAR_LOG_HIGH))
    f = Uniform(t(PRIOR_LOW[2]), t(PRIOR_HIGH[2]))
    return [d, log_dstar, f]


def get_alt_processed_prior(device: str = "cpu", log_dstar: bool = True,
                            dstar_log_mu: float = ALT_DSTAR_LOG_MU,
                            dstar_log_sigma: float = ALT_DSTAR_LOG_SIGMA):
    """sbi triple ``(prior, num_parameters, prior_returns_numpy)`` for the
    tissue-informed prior. Requires ``log_dstar=True`` (the D* marginal is on
    log10(D*)); a False value is a usage error and raises.
    """
    if not log_dstar:
        raise ValueError(
            "npe_prior_alt is defined in log10(D*) space; train it with --log-dstar.")
    return process_prior(_component_priors(device=device,
                                           dstar_log_mu=dstar_log_mu,
                                           dstar_log_sigma=dstar_log_sigma))
