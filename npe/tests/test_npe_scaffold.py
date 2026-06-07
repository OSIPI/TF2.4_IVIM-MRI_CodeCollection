"""
test_npe_scaffold.py
====================
Invariants for the Phase-E NPE scaffold (Step 4). These guard the contracts the
later training step depends on; none of them trains a network.

Cases
-----
1. prior support bounds (samples in-box; log_prob finite inside / -inf outside).
2. masked-grid simulator output shape + finiteness + mask semantics.
3. set simulator output shape + finiteness + (b, S) layout.
4. embedding output is invariant to set ORDERING (permutation invariance).
5. embedding output DIMENSION is invariant to set LENGTH.
6. clean and noisy modes differ ONLY by the Rician noise term.
"""
import numpy as np
import torch

from ivim_simulator import B_SCHEMES
from npe_prior import (
    N_PARAMS,
    PRIOR_HIGH,
    PRIOR_LOW,
    get_prior,
    get_processed_prior,
)
from npe_simulator import IVIMNPESimulator, build_set_embedding

# A few fixed, in-box truths [D, Dstar, f] (absolute units) for deterministic tests.
THETA = torch.tensor(
    [
        [0.7e-3, 30.0e-3, 0.20],
        [1.3e-3, 50.0e-3, 0.30],
        [2.0e-3, 10.0e-3, 0.05],
        [0.2e-3, 3.0e-3, 0.00],   # corner: f -> 0 (mono-exp), Dstar -> low
    ],
    dtype=torch.float32,
)


def test_prior_support_bounds():
    prior, n_params, _ = get_processed_prior()
    assert n_params == N_PARAMS == 3

    low = torch.as_tensor(PRIOR_LOW)
    high = torch.as_tensor(PRIOR_HIGH)

    samples = prior.sample((5000,))
    assert samples.shape == (5000, 3)
    assert (samples >= low).all() and (samples <= high).all()

    p = get_prior()
    inside = ((low + high) / 2).unsqueeze(0)
    assert torch.isfinite(p.log_prob(inside)).all()
    # Each dimension pushed just outside its bound -> zero density (-inf log_prob).
    for i in range(3):
        bad = inside.clone()
        bad[0, i] = high[i] + 1.0
        assert torch.isinf(p.log_prob(bad)).all()


def test_simulator_masked_grid_shape():
    sim = IVIMNPESimulator(representation="masked_grid", clean=False, seed=0)
    obs, snr_ctx = sim(THETA)

    assert obs.shape == (THETA.shape[0], 2 * sim.k_super)
    assert snr_ctx.shape == (THETA.shape[0], 1)
    assert torch.isfinite(obs).all() and torch.isfinite(snr_ctx).all()

    # Second half is the validity mask; its column sum equals the active count.
    mask = obs[:, sim.k_super:]
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert int(mask[0].sum().item()) == sim.k_active
    # Signal entries at masked-out positions must be exactly zero.
    signal = obs[:, : sim.k_super]
    assert torch.equal(signal[mask == 0.0], torch.zeros_like(signal[mask == 0.0]))


def test_simulator_set_shape():
    sim = IVIMNPESimulator(representation="set", clean=False, seed=0)
    obs, snr_ctx = sim(THETA)

    assert obs.shape == (THETA.shape[0], sim.k_active, 2)
    assert snr_ctx.shape == (THETA.shape[0], 1)
    assert torch.isfinite(obs).all() and torch.isfinite(snr_ctx).all()

    # Channel 0 is the b-value (shared across the batch); channel 1 is the signal.
    b_channel = obs[:, :, 0]
    expected_b = torch.as_tensor(sim.active_bvals, dtype=torch.float32)
    assert torch.allclose(b_channel[0], expected_b)
    assert torch.allclose(b_channel, b_channel[:1].expand_as(b_channel))


def test_embedding_permutation_invariance():
    sim = IVIMNPESimulator(representation="set", clean=False, seed=1)
    obs, _ = sim(THETA)

    embedding = build_set_embedding(output_dim=20, seed=0)
    embedding.eval()
    perm = torch.randperm(obs.shape[1])
    with torch.no_grad():
        base = embedding(obs)
        shuffled = embedding(obs[:, perm, :])

    assert base.shape == (THETA.shape[0], 20)
    assert torch.allclose(base, shuffled, atol=1e-4)


def test_embedding_length_invariance():
    # Two acquisitions of different length feed the SAME embedding -> same out dim.
    sim_short = IVIMNPESimulator(representation="set",
                                 active_bvals=B_SCHEMES["clinical_sparse"], seed=0)
    sim_long = IVIMNPESimulator(representation="set",
                                active_bvals=B_SCHEMES["dense"], seed=0)
    obs_short, _ = sim_short(THETA)
    obs_long, _ = sim_long(THETA)
    assert obs_short.shape[1] != obs_long.shape[1]  # genuinely different lengths

    embedding = build_set_embedding(output_dim=20, seed=0)
    embedding.eval()
    with torch.no_grad():
        emb_short = embedding(obs_short)
        emb_long = embedding(obs_long)

    assert emb_short.shape == (THETA.shape[0], 20)
    assert emb_long.shape == (THETA.shape[0], 20)
    assert emb_short.shape[-1] == emb_long.shape[-1]


def test_clean_vs_noisy_differ_only_by_noise():
    # Identical config except the clean flag; pin SNR and RNG so the only possible
    # difference is the Rician noise term.
    common = dict(representation="masked_grid", active_bvals=B_SCHEMES["clinical_sparse"])
    sim_clean = IVIMNPESimulator(clean=True, **common)
    sim_noisy = IVIMNPESimulator(clean=False, **common)

    obs_clean, ctx_clean = sim_clean(THETA, snr=1e12,
                                     rng=np.random.default_rng(0))
    # SNR -> infinity makes sigma -> 0, so the Rician term vanishes: noisy == clean.
    obs_noisy_hi, ctx_noisy_hi = sim_noisy(THETA, snr=1e12,
                                           rng=np.random.default_rng(0))
    assert torch.allclose(obs_clean, obs_noisy_hi, atol=1e-5)
    # The SNR context is computed identically regardless of the clean flag.
    assert torch.allclose(ctx_clean, ctx_noisy_hi)

    # At a realistic SNR the noise term is present, so the outputs must differ.
    obs_noisy_lo, _ = sim_noisy(THETA, snr=20.0, rng=np.random.default_rng(0))
    assert not torch.allclose(obs_clean, obs_noisy_lo, atol=1e-3)
