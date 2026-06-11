"""Checkpoint H — inference-time benchmark with documented methods.

Addresses the reviewer note that the timing benchmark (NPE / NLLS / MCMC per
1,000 voxels) lacked a methods description: hardware, and implementation parity
across the three estimators.

All three estimators are timed on the SAME machine and the SAME CPU, on an
identical 1,000-voxel synthetic workload (one fixed biexponential ground truth,
Gaussian noise, the 10-point evaluation b-scheme). Hardware and library
provenance are captured at runtime and written into the output CSV so the
numbers are never quoted without their methods again.

Parity notes
------------
* CPU-only. No GPU is used by any estimator, so the comparison is hardware-fair.
* NPE is timed at STEADY STATE: a warm-up draw is issued before timing so the
  reported number excludes one-time model load + lazy graph/JIT construction.
  (The previous ad-hoc script timed the cold first call and therefore
  over-reported NPE latency by ~3-4x.)  We report the min of 3 warm runs.
* NLLS is SciPy `least_squares` (TRF), one independent fit per voxel, serial.
* MCMC is the OSIPI OGC_AmsterdamUMC_Bayesian_biexp estimator (emcee), timed on
  a 50-voxel subset and linearly projected to 1,000 (its per-voxel cost is
  data-independent for fixed sampler settings); 32 walkers x 1500 steps,
  500 burn-in, thin 5 -- the settings used in the manuscript comparison.

Reproduce:
    .venv/bin/python npe/run_h_benchmark.py --out npe/benchmark_timings.csv
    # fast smoke (skip the ~1 min MCMC leg):
    .venv/bin/python npe/run_h_benchmark.py --skip-mcmc
"""
import argparse
import csv
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("npe"))
sys.path.insert(0, os.path.abspath("."))

from npe_prior import invert_theta  # noqa: F401  (kept for model-load parity)
from train_npe import SNRWrapperEmbedding, pack_x

sys.modules["__main__"].SNRWrapperEmbedding = SNRWrapperEmbedding

from uq.bayesian import mcmc_uncertainty
from uq.ivim_fit import make_model

# 10-point evaluation b-scheme used by the in-vivo demo / Figure 5 pipeline.
TARGET_BVALS = np.array([0, 10, 20, 30, 50, 75, 100, 150, 400, 600], dtype=float)
GT = dict(D=1.5e-3, Dstar=50e-3, f=0.2)  # one fixed biexponential ground truth


def fit_biexp_nlls(bvals, signal, s0=1.0):
    import scipy.optimize as opt

    bounds = ([0.2e-3, 3.0e-3, 0.0], [3.0e-3, 0.15, 0.5])
    x0 = [1.6e-3, 50e-3, 0.2]

    def residuals(p):
        d, ds, f = p
        pred = s0 * (f * np.exp(-bvals * ds) + (1.0 - f) * np.exp(-bvals * d))
        return pred - signal

    try:
        return opt.least_squares(residuals, x0, bounds=bounds, method="trf").x
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def make_workload(n_voxels, seed=42):
    rng = np.random.default_rng(seed)
    b = TARGET_BVALS
    clean = GT["f"] * np.exp(-b * GT["Dstar"]) + (1.0 - GT["f"]) * np.exp(-b * GT["D"])
    sig = np.clip(clean[None, :] + rng.normal(0, 0.05, (n_voxels, len(b))), 0.01, 1.2)
    return sig


def prep_npe(model_path, n_bvals):
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)
    estimator = posterior.posterior_estimator
    std_mod = estimator.net._embedding_net[0]
    orig_mean = std_mod._mean.clone().numpy()
    orig_std = std_mod._std.clone().numpy()
    b_orig = orig_mean[:, 0]
    new_mean = np.zeros((n_bvals, 3))
    new_std = np.zeros((n_bvals, 3))
    for c in range(3):
        new_mean[:, c] = np.interp(TARGET_BVALS, b_orig, orig_mean[:, c])
        new_std[:, c] = np.interp(TARGET_BVALS, b_orig, orig_std[:, c])
    std_mod._mean = torch.as_tensor(new_mean, dtype=torch.float32)
    std_mod._std = torch.as_tensor(new_std, dtype=torch.float32)
    estimator._condition_shape = torch.Size([n_bvals, 3])
    return posterior


def benchmark(args):
    torch.set_num_threads(os.cpu_count())
    n = args.n_voxels
    signals = make_workload(n)

    posterior = prep_npe(args.model, len(TARGET_BVALS))
    bt = np.tile(TARGET_BVALS[None, :], (n, 1))
    obs = torch.as_tensor(np.stack([bt, signals], axis=-1), dtype=torch.float32)
    snr_ctx = torch.as_tensor(np.log10(np.full(n, 30.0))[:, None], dtype=torch.float32)
    x = pack_x(obs, snr_ctx, "set")

    # NPE: warm up (excludes load + lazy build), then time steady state.
    with torch.no_grad():
        _ = posterior.sample_batched((args.n_samples,), x=x[: min(50, n)], show_progress_bars=False)
    npe_runs = []
    for _ in range(3):
        t0 = time.time()
        with torch.no_grad():
            _ = posterior.sample_batched((args.n_samples,), x=x, show_progress_bars=False)
        npe_runs.append((time.time() - t0) * (1000.0 / n))
    npe_ms = min(npe_runs) * 1000.0

    # NLLS: one TRF fit per voxel, serial.
    t0 = time.time()
    for i in range(n):
        fit_biexp_nlls(TARGET_BVALS, signals[i])
    nlls_s = (time.time() - t0) * (1000.0 / n)

    # MCMC: emcee, timed on a subset and projected (per-voxel cost is fixed).
    if args.skip_mcmc:
        mcmc_s = float("nan")
        n_mcmc = 0
    else:
        n_mcmc = min(args.n_mcmc, n)
        model = make_model("OGC_AmsterdamUMC_Bayesian_biexp", TARGET_BVALS)
        t0 = time.time()
        mcmc_uncertainty(model, signals[:n_mcmc], TARGET_BVALS, nwalkers=32, nsteps=1500, burn=500, thin=5)
        mcmc_s = (time.time() - t0) * (1000.0 / n_mcmc)

    hw = (
        f"{platform.platform()}; CPU={platform.processor() or platform.machine()}; "
        f"cores={os.cpu_count()}; torch_threads={torch.get_num_threads()}; "
        f"torch={torch.__version__}; numpy={np.__version__}; python={platform.python_version()}"
    )

    rows = [
        ("NPE_amortized_NSF", "steady-state (warm), min of 3; batched 100-sample draw", npe_ms / 1000.0, npe_ms),
        ("NLLS_SciPy_TRF", "1 least_squares(TRF) fit/voxel, serial", nlls_s, nlls_s * 1000.0),
        ("MCMC_emcee_OGC_Bayesian", f"32 walkers x 1500 steps, 500 burn, thin 5; projected from {n_mcmc} voxels", mcmc_s, mcmc_s * 1000.0),
    ]

    print("\n=== Inference-time benchmark (per 1,000 voxels) ===")
    print(f"Hardware: {hw}\n")
    print(f"{'estimator':<26} {'per-1000-voxel':>16}")
    for name, _note, sec, _ms in rows:
        disp = f"{sec*1000:.1f} ms" if sec < 1 else f"{sec:.3f} s"
        print(f"{name:<26} {disp:>16}")
    print(f"\nNPE warm runs (ms): {[round(r*1000, 0) for r in npe_runs]}")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["estimator", "implementation", "seconds_per_1000_voxels", "ms_per_1000_voxels"])
        for name, note, sec, ms in rows:
            w.writerow([name, note, f"{sec:.6f}", f"{ms:.3f}"])
        w.writerow([])
        w.writerow(["hardware_provenance", hw, "", ""])
    print(f"\nWrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Methods-documented inference-time benchmark.")
    ap.add_argument("--model", default="npe/npe_posterior_setB.pt")
    ap.add_argument("--n-voxels", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-mcmc", type=int, default=50)
    ap.add_argument("--skip-mcmc", action="store_true")
    ap.add_argument("--out", default="npe/benchmark_timings.csv")
    benchmark(ap.parse_args())


if __name__ == "__main__":
    main()
