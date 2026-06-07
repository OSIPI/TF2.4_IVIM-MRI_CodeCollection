"""
run_e_smoke.py
==============
Fast, no-training smoke check for the Phase-E NPE scaffold (Step 3).

It draws ~1000 prior samples, runs the simulator in every (clean/noisy) x
(masked_grid/set) mode, asserts finite outputs of the right shape, pushes one
batch through a ``PermutationInvariantEmbedding`` to confirm its output dimension,
and measures the per-simulation cost so the training-simulation budget can be
sized next. There is deliberately NO ``.train()`` call here.

Run:
    PYTHONPATH=. ../.venv-npe/bin/python run_e_smoke.py
    # or, from the repo root:
    PYTHONPATH=npe .venv-npe/bin/python npe/run_e_smoke.py
"""
from __future__ import annotations

import time

import numpy as np
import torch

from npe_prior import (
    DISPLAY_UNITS,
    PARAM_NAMES,
    PRIOR_HIGH,
    PRIOR_LOW,
    get_processed_prior,
    to_display,
)
from npe_simulator import IVIMNPESimulator, build_set_embedding

N_SAMPLES = 1000
EMBED_OUTPUT_DIM = 20
SEED = 0


def _check_finite(name, t: torch.Tensor):
    if not torch.isfinite(t).all():
        raise AssertionError(f"{name}: non-finite values in output")


def main() -> None:
    torch.manual_seed(SEED)
    print("=" * 70)
    print("Phase-E NPE scaffold smoke test")
    print("=" * 70)

    # -- Step: prior ------------------------------------------------------- #
    prior, n_params, returns_numpy = get_processed_prior()
    print(f"\n[prior] process_prior -> n_params={n_params}, "
          f"returns_numpy={returns_numpy}")
    assert n_params == 3, "expected a 3-D prior [D, Dstar, f]"

    theta = prior.sample((N_SAMPLES,))
    assert theta.shape == (N_SAMPLES, 3), theta.shape
    low = torch.as_tensor(PRIOR_LOW)
    high = torch.as_tensor(PRIOR_HIGH)
    assert (theta >= low).all() and (theta <= high).all(), "prior sample out of box"

    disp = to_display(theta)
    print(f"[prior] drew {N_SAMPLES} samples; per-dimension stats:")
    print(f"  {'param':<6} {'abs min':>10} {'abs max':>10} {'abs mean':>10} "
          f"  | display (min / max / mean)  [{', '.join(DISPLAY_UNITS)}]")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:<6} {theta[:, i].min():10.3e} {theta[:, i].max():10.3e} "
              f"{theta[:, i].mean():10.3e}   | "
              f"{disp[:, i].min():8.3f} / {disp[:, i].max():8.3f} / "
              f"{disp[:, i].mean():8.3f}")

    # -- Step: simulator in every mode ------------------------------------- #
    print(f"\n[simulator] running all 4 modes on {N_SAMPLES} draws...")
    per_sim_us = {}
    for representation in ("masked_grid", "set"):
        for clean in (True, False):
            sim = IVIMNPESimulator(representation=representation, clean=clean,
                                   seed=SEED)
            t0 = time.perf_counter()
            obs, snr_ctx = sim(theta)
            dt = time.perf_counter() - t0

            tag = f"{representation}/{'clean' if clean else 'noisy'}"
            exp_obs = (N_SAMPLES,) + sim.observation_shape
            assert tuple(obs.shape) == exp_obs, (tag, obs.shape, exp_obs)
            assert tuple(snr_ctx.shape) == (N_SAMPLES, 1), (tag, snr_ctx.shape)
            _check_finite(f"{tag} obs", obs)
            _check_finite(f"{tag} snr_ctx", snr_ctx)

            us = dt / N_SAMPLES * 1e6
            per_sim_us[tag] = us
            print(f"  {tag:<22} obs={tuple(obs.shape)}  snr_ctx={tuple(snr_ctx.shape)}"
                  f"  ({us:6.2f} us/sim)")

    # -- Step: permutation-invariant embedding ----------------------------- #
    print("\n[embedding] PermutationInvariantEmbedding over (b_i, S_i) sets...")
    set_sim = IVIMNPESimulator(representation="set", clean=False, seed=SEED)
    set_obs, _ = set_sim(theta)                       # (N, K_active, 2)
    embedding = build_set_embedding(output_dim=EMBED_OUTPUT_DIM, seed=SEED)
    embedding.eval()
    with torch.no_grad():
        emb = embedding(set_obs)                      # (N, EMBED_OUTPUT_DIM)
    _check_finite("embedding", emb)
    assert emb.shape == (N_SAMPLES, EMBED_OUTPUT_DIM), emb.shape
    out_dim = emb.shape[-1]
    print(f"  input set shape {tuple(set_obs.shape)} -> embedding {tuple(emb.shape)} "
          f"(output_dim={out_dim})")

    # quick invariance sanity (shuffle the trial axis -> identical embedding)
    perm = torch.randperm(set_obs.shape[1])
    with torch.no_grad():
        emb_perm = embedding(set_obs[:, perm, :])
    max_dev = (emb - emb_perm).abs().max().item()
    print(f"  permutation-invariance check: max|Δ| over shuffled b-order = "
          f"{max_dev:.2e}")
    assert max_dev < 1e-4, "embedding is not permutation invariant"

    # -- summary ----------------------------------------------------------- #
    slowest = max(per_sim_us.values())
    print("\n" + "-" * 70)
    print(f"per-simulation cost (slowest mode): {slowest:.2f} us/sim "
          f"-> {slowest * 1e-6:.2e} s/sim")
    print(f"=> ~{1e6 / slowest:,.0f} sims/sec single-process; "
          f"1e6 training sims ~= {1e6 * slowest * 1e-6:.1f} s")
    print("-" * 70)
    print("scaffold OK")


if __name__ == "__main__":
    main()
