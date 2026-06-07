"""
npe_simulator.py
================
sbi-compatible simulator ``theta -> x`` for the Phase-E IVIM NPE scaffold
(Step 1). It wraps the pure-numpy forward model in ``ivim_simulator.ivim_signal``,
runs batched, and emits an ``(observation, snr_context)`` pair per draw.

theta = ``[D, Dstar, f]`` in absolute units (mm^2/s, mm^2/s, unitless); see
``npe_prior`` for the box and the display-scaling convention.

Two orthogonal, flag-selectable axes
------------------------------------
1. ``clean`` (bool):
     * ``clean=True``  -> noise-free signal (later: calibration ground truth).
     * ``clean=False`` -> Rician-corrupted signal (later: training data).
   The two paths are *identical except for the noise term* -- noisy simply pipes
   the clean signal through ``ivim_simulator.add_rician_noise``.

2. ``representation`` (str):
     * ``"masked_grid"`` -> a fixed-length vector on a SUPERSET b-grid with an
       explicit validity mask, for the simple-MLP pipeline-check path. Shape
       ``(B, 2 * K_super)`` = ``[masked_signal | mask]``.
     * ``"set"``         -> the per-acquisition set of ``(b_i, S_i)`` pairs, for
       sbi's ``PermutationInvariantEmbedding``. Shape ``(B, K_active, 2)``. This
       is the principled path for amortizing over heterogeneous external b-grids.

SNR is *known context*, not a free theta dimension
--------------------------------------------------
A scalar SNR is drawn per simulation (log-uniform over a configurable range by
default) and emitted as ``snr_context`` shaped ``(B, 1)`` carrying ``log10(SNR)``
-- a network-friendly encoding of a positive, scale-spanning quantity. The atlas
is SNR-resolved by design: downstream the context is concatenated to the MLP input
or fed alongside the set embedding. ``clean=True`` still draws and reports an SNR
(so contexts line up across modes); it just isn't applied to the signal.

All public outputs are ``torch.float32`` tensors so they drop straight into sbi.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from ivim_simulator import B_SCHEMES, add_rician_noise, ivim_signal

# --------------------------------------------------------------------------- #
# b-value grids
# --------------------------------------------------------------------------- #
# The SUPERSET grid is the union of every acquisition scheme the existing study
# uses; the masked-grid representation lives on it and a per-acquisition mask
# selects the active subset. Building it from B_SCHEMES keeps the NPE scaffold in
# lockstep with the rest of the project rather than hard-coding a second list.
B_SUPERSET = np.unique(np.concatenate([np.asarray(v, float)
                                       for v in B_SCHEMES.values()]))
# Default acquisition actually "measured" per draw -- the clinical sparse scheme,
# a strict subset of the superset (so the mask is meaningful, not all-ones).
DEFAULT_ACTIVE_BVALS = np.asarray(B_SCHEMES["clinical_sparse"], float)

# Default SNR sampling range (defined on S0, i.e. at b=0). Spans a hard,
# noise-floor-dominated regime up to an easy one; configurable per simulator.
SNR_MIN = 8.0
SNR_MAX = 100.0

REPRESENTATIONS = ("masked_grid", "set")


def _as_2d_theta(theta) -> np.ndarray:
    """Coerce theta (torch/numpy/list) to a contiguous float64 ``(B, 3)`` array."""
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    if theta.shape[-1] != 3:
        raise ValueError(f"theta must have last dim 3 [D, Dstar, f], got {theta.shape}")
    return theta


class IVIMNPESimulator:
    """Batched, sbi-compatible IVIM simulator emitting ``(observation, snr_context)``.

    Parameters
    ----------
    representation : {"masked_grid", "set"}
        Observation layout (see module docstring).
    clean : bool
        If True, return the noise-free signal; else Rician-corrupted.
    active_bvals : array-like, optional
        b-values actually acquired per draw (s/mm^2). Defaults to the clinical
        sparse scheme. Any value not already in ``b_superset`` is folded into the
        superset so ``active_bvals`` is always a subset of it.
    b_superset : array-like, optional
        Superset grid for the masked representation. Defaults to ``B_SUPERSET``.
    snr_range : (float, float)
        Inclusive ``(min, max)`` SNR sampling bounds.
    snr_log : bool
        If True (default) sample SNR log-uniformly across ``snr_range``; else
        uniformly.
    S0 : float
        Signal at b=0 (normalized to 1.0 by convention).
    seed : int, optional
        Seed for the internal RNG used for SNR draws and Rician noise.
    """

    def __init__(
        self,
        *,
        representation: str = "masked_grid",
        clean: bool = False,
        active_bvals=None,
        b_superset=None,
        snr_range: Tuple[float, float] = (SNR_MIN, SNR_MAX),
        snr_log: bool = True,
        S0: float = 1.0,
        seed: Optional[int] = None,
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {representation!r}"
            )
        self.representation = representation
        self.clean = bool(clean)
        self.S0 = float(S0)
        self.snr_log = bool(snr_log)

        lo, hi = float(snr_range[0]), float(snr_range[1])
        if not (0.0 < lo <= hi):
            raise ValueError(f"snr_range must satisfy 0 < min <= max, got {snr_range}")
        self.snr_range = (lo, hi)

        active = (DEFAULT_ACTIVE_BVALS if active_bvals is None
                  else np.asarray(active_bvals, dtype=float).ravel())
        superset = (B_SUPERSET if b_superset is None
                    else np.asarray(b_superset, dtype=float).ravel())
        # Guarantee active is a subset of the superset, then sort/unique both.
        superset = np.unique(np.concatenate([superset, active]))
        self.b_superset = superset
        self.active_bvals = np.unique(active)
        # Boolean validity mask + integer positions of the active b's in superset.
        self.active_mask = np.isin(self.b_superset, self.active_bvals)
        self.k_super = int(self.b_superset.size)
        self.k_active = int(self.active_bvals.size)

        self.rng = np.random.default_rng(seed)

    # -- shape introspection -------------------------------------------------- #
    @property
    def observation_shape(self) -> tuple:
        """Per-sample observation shape (excluding the leading batch axis)."""
        if self.representation == "masked_grid":
            return (2 * self.k_super,)
        return (self.k_active, 2)

    # -- building blocks ------------------------------------------------------ #
    def sample_snr(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw ``n`` scalar SNRs over ``snr_range`` (log-uniform by default)."""
        rng = self.rng if rng is None else rng
        lo, hi = self.snr_range
        if self.snr_log:
            return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), size=n)
        return rng.uniform(lo, hi, size=n)

    def clean_signals(self, theta, bvals: np.ndarray) -> np.ndarray:
        """Noise-free IVIM signal for every theta row over ``bvals`` -> ``(B, K)``.

        Pure and deterministic -- the shared core of both clean and noisy modes.
        """
        theta = _as_2d_theta(theta)
        D, Dstar, f = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]
        bvals = np.asarray(bvals, dtype=float)[None, :]
        return ivim_signal(bvals, D, Dstar, f, S0=self.S0)  # (B, K)

    # -- the simulator -------------------------------------------------------- #
    def __call__(
        self,
        theta,
        *,
        snr=None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map ``theta -> (observation, snr_context)``.

        Parameters
        ----------
        theta : tensor/array ``(B, 3)`` (or ``(3,)``)
            ``[D, Dstar, f]`` in absolute units.
        snr : float or array ``(B,)``, optional
            Pin the SNR(s) instead of sampling -- used by tests/calibration.
        rng : numpy Generator, optional
            Override the internal RNG for this call (reproducibility).

        Returns
        -------
        observation : torch.float32, shape ``(B, 2*K_super)`` (masked_grid) or
            ``(B, K_active, 2)`` (set).
        snr_context : torch.float32, shape ``(B, 1)`` carrying ``log10(SNR)``.
        """
        rng = self.rng if rng is None else rng
        theta = _as_2d_theta(theta)
        n = theta.shape[0]

        if snr is None:
            snr_vec = self.sample_snr(n, rng=rng)
        else:
            snr_vec = np.broadcast_to(np.asarray(snr, dtype=float), (n,)).copy()
        snr_col = snr_vec[:, None]  # (B, 1) for per-row broadcasting

        if self.representation == "masked_grid":
            clean = self.clean_signals(theta, self.b_superset)            # (B, K_super)
            signal = clean if self.clean else add_rician_noise(
                clean, snr_col, S0=self.S0, rng=rng)
            mask = self.active_mask.astype(np.float64)[None, :]           # (1, K_super)
            masked_signal = signal * mask                                 # zero out inactive
            mask_tiled = np.broadcast_to(mask, masked_signal.shape)
            obs = np.concatenate([masked_signal, mask_tiled], axis=1)     # (B, 2*K_super)
        else:  # "set"
            clean = self.clean_signals(theta, self.active_bvals)          # (B, K_active)
            signal = clean if self.clean else add_rician_noise(
                clean, snr_col, S0=self.S0, rng=rng)
            b_tiled = np.broadcast_to(self.active_bvals[None, :], signal.shape)
            obs = np.stack([b_tiled, signal], axis=-1)                    # (B, K_active, 2)

        observation = torch.as_tensor(np.ascontiguousarray(obs), dtype=torch.float32)
        snr_context = torch.as_tensor(np.log10(snr_col), dtype=torch.float32)  # (B, 1)
        return observation, snr_context


# --------------------------------------------------------------------------- #
# Set-embedding helper (representation (b)): one place to build the net so the
# smoke test and the pytest suite configure it identically.
# --------------------------------------------------------------------------- #
def build_set_embedding(
    *,
    trial_feature_dim: int = 2,
    latent_dim: int = 16,
    output_dim: int = 20,
    num_layers_trial: int = 2,
    num_hiddens_trial: int = 40,
    num_layers_out: int = 2,
    num_hiddens_out: int = 40,
    aggregation_fn: str = "mean",
    seed: Optional[int] = None,
):
    """A ``PermutationInvariantEmbedding`` over ``(b_i, S_i)`` set observations.

    Each trial (one b-value measurement) is a ``(b, S)`` pair embedded by an
    ``FCEmbedding`` to ``latent_dim``; the per-trial embeddings are aggregated
    (mean -> permutation- and trial-count-invariant) and an output MLP maps to a
    fixed ``output_dim``. Input shape ``(B, K, trial_feature_dim)`` -> output
    ``(B, output_dim)``, regardless of K or its ordering.
    """
    from sbi.neural_nets.embedding_nets import (
        FCEmbedding,
        PermutationInvariantEmbedding,
    )

    if seed is not None:
        torch.manual_seed(seed)
    trial_net = FCEmbedding(
        input_dim=trial_feature_dim,
        output_dim=latent_dim,
        num_layers=num_layers_trial,
        num_hiddens=num_hiddens_trial,
    )
    return PermutationInvariantEmbedding(
        trial_net,
        trial_net_output_dim=latent_dim,
        aggregation_fn=aggregation_fn,
        num_layers=num_layers_out,
        num_hiddens=num_hiddens_out,
        output_dim=output_dim,
        aggregation_dim=1,
    )
