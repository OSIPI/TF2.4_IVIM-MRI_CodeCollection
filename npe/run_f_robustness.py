"""
run_f_robustness.py
===================
Phase F: Misspecification Stress Test (Checkpoint F1).
Tests whether the NPE posterior under-covers held-out b-value signals under model misspecification
(tri-exponential truth and Gaussian noise) compared to NLLS. Also evaluates efficiency on an
alternative b-scheme to test the amortization claim.
"""
from __future__ import annotations

import os
import sys
import time
import csv
import argparse
import multiprocessing
import numpy as np
import scipy.optimize as opt
import torch

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Handle imports robustly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_prior import get_processed_prior, DISPLAY_UNITS, DISPLAY_SCALE, to_display, invert_theta
from npe_simulator import IVIMNPESimulator, B_SCHEMES
from train_npe import pack_x, SNRWrapperEmbedding


# --------------------------------------------------------------------------- #
# Fit and CRLB helpers (copied from validation/efficiency scripts for safety)
# --------------------------------------------------------------------------- #

def compute_jacobian(theta: np.ndarray, bvals: np.ndarray, S0: float = 1.0) -> np.ndarray:
    """
    Computes the Jacobian of the IVIM biexponential signal model w.r.t. theta=[D, Dstar, f].
    Shape: (K, 3) where K is the number of b-values.
    """
    D, Dstar, f = theta[0], theta[1], theta[2]
    bvals = np.asarray(bvals, dtype=float)
    
    dS_dD = -S0 * (1.0 - f) * bvals * np.exp(-bvals * D)
    dS_dDstar = -S0 * f * bvals * np.exp(-bvals * Dstar)
    dS_df = S0 * (np.exp(-bvals * Dstar) - np.exp(-bvals * D))
    
    return np.stack([dS_dD, dS_dDstar, dS_df], axis=1)


def compute_crlb(theta: np.ndarray, bvals: np.ndarray, snr: float, S0: float = 1.0) -> np.ndarray:
    """
    Computes analytical standard deviation lower bounds (sqrt_crlb) for [D, Dstar, f] in absolute units.
    """
    J = compute_jacobian(theta, bvals, S0)
    sigma = S0 / snr
    FIM = (J.T @ J) / (sigma ** 2)
    try:
        CRLB_cov = np.linalg.inv(FIM)
        var = np.diag(CRLB_cov)
        if np.any(var < 0):
            return np.array([np.nan, np.nan, np.nan])
        return np.sqrt(var)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan])


def fit_biexp_nlls(bvals: np.ndarray, signal: np.ndarray, S0: float = 1.0) -> np.ndarray:
    """
    Performs a bounded non-linear least squares fit to biexponential model.
    Bounds match prior: D in [0.2e-3, 3.0e-3], Dstar in [3.0e-3, 0.15], f in [0.0, 0.5].
    """
    bounds = (
        [0.2e-3, 3.0e-3, 0.0],  # lower
        [3.0e-3, 0.15, 0.5]     # upper
    )
    x0 = [1.6e-3, 50e-3, 0.2]
    
    def residuals(params):
        D, Dstar, f = params
        pred = S0 * (f * np.exp(-bvals * Dstar) + (1.0 - f) * np.exp(-bvals * D))
        return pred - signal
    
    try:
        res = opt.least_squares(residuals, x0, bounds=bounds, method="trf")
        return res.x
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def _nlls_worker(args):
    bvals, signal = args
    return fit_biexp_nlls(bvals, signal)


# --------------------------------------------------------------------------- #
# OOD Generators
# --------------------------------------------------------------------------- #

def biexp_signal(bvals, D, Dstar, f, S0=1.0):
    """Clean biexponential signal."""
    bvals = np.asarray(bvals, dtype=float)
    return S0 * (f * np.exp(-bvals * Dstar) + (1.0 - f) * np.exp(-bvals * D))


def triexp_signal(bvals, D, Dstar, f, S0=1.0):
    """
    OOD tri-exponential model. Splits the perfusion rate Dstar and fraction f:
    f1 = 0.6*f, f2 = 0.4*f
    Dstar1 = 0.5*Dstar, Dstar2 = 2.0*Dstar
    """
    bvals = np.asarray(bvals, dtype=float)
    f1, f2 = 0.6 * f, 0.4 * f
    Dstar1, Dstar2 = 0.5 * Dstar, 2.0 * Dstar
    return S0 * (f1 * np.exp(-bvals * Dstar1) + f2 * np.exp(-bvals * Dstar2) + (1.0 - f1 - f2) * np.exp(-bvals * D))


def add_rician_noise(signal, snr, S0=1.0, rng=None):
    """Standard Rician noise."""
    rng = np.random.default_rng() if rng is None else rng
    sigma = S0 / snr
    signal = np.asarray(signal, dtype=float)
    n_real = rng.normal(0.0, sigma, size=signal.shape)
    n_imag = rng.normal(0.0, sigma, size=signal.shape)
    return np.sqrt((signal + n_real) ** 2 + n_imag ** 2)


def add_gaussian_noise(signal, snr, S0=1.0, rng=None):
    """Gaussian noise (zero-mean, noise floor misspecification)."""
    rng = np.random.default_rng() if rng is None else rng
    sigma = S0 / snr
    signal = np.asarray(signal, dtype=float)
    return signal + rng.normal(0.0, sigma, size=signal.shape)


# --------------------------------------------------------------------------- #
# Asymptotic Covariance Sampling for NLLS
# --------------------------------------------------------------------------- #

def sample_nlls_asymptotic(theta_fit, bvals_fit, snr, num_samples=200, bounds=None, rng=None):
    """
    Draws parameter samples for NLLS using the Fisher Information Matrix covariance.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    D, Dstar, f = theta_fit[0], theta_fit[1], theta_fit[2]
    if np.isnan(D) or np.isnan(Dstar) or np.isnan(f):
        # Draw uniform samples if fit failed
        if bounds is not None:
            return rng.uniform(bounds[0], bounds[1], size=(num_samples, 3))
        return np.zeros((num_samples, 3))
    
    J = compute_jacobian(theta_fit, bvals_fit, S0=1.0)
    sigma = 1.0 / snr
    FIM = (J.T @ J) / (sigma ** 2)
    try:
        cov = np.linalg.inv(FIM)
        cov += 1e-12 * np.eye(3)  # add ridge for numerical stability
        samples = rng.multivariate_normal(theta_fit, cov, size=num_samples)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to a small diagonal covariance
        diag_var = np.array([0.5e-3, 20e-3, 0.1]) ** 2
        cov = np.diag(diag_var) / (snr ** 2)
        samples = rng.multivariate_normal(theta_fit, cov, size=num_samples)
        
    if bounds is not None:
        samples = np.clip(samples, bounds[0], bounds[1])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Checkpoint F1 Robustness Stress Test.")
    parser.add_argument("--model", type=str, default="npe/npe_posterior_setB.pt", help="Path to model file.")
    parser.add_argument("--suffix", type=str, default="_f1", help="Suffix for output files.")
    parser.add_argument("--n-repeats", type=int, default=50, help="Number of noise realizations per grid point.")
    parser.add_argument("--n-noisy-samples", type=int, default=200, help="Number of posterior samples to draw.")
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("Checkpoint F1: Misspecification Stress Test")
    print("=" * 80)

    # 1. Load Model
    model_path = args.model
    if not os.path.exists(model_path):
        alt_path = "npe/" + os.path.basename(model_path) if not model_path.startswith("npe/") else os.path.basename(model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find model at {model_path}")

    sys.modules['__main__'].SNRWrapperEmbedding = SNRWrapperEmbedding
    print(f"Loading setB model from {model_path}...")
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)

    # Save original standardize mean and std to support dynamic set size adjustments
    estimator = posterior.posterior_estimator
    std_mod = estimator.net._embedding_net[0]
    orig_mean = std_mod._mean.clone()
    orig_std = std_mod._std.clone()

    def set_npe_condition_size(K: int):
        estimator._condition_shape = torch.Size([K, 3])
        std_mod._mean = orig_mean[:K, :]
        std_mod._std = orig_std[:K, :]

    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    print(f"Auto-detected log_dstar = {log_dstar}")

    prior, _, _ = get_processed_prior(device="cpu", log_dstar=log_dstar)
    prior_bounds = (
        [0.2e-3, 3.0e-3, 0.0],  # lower
        [3.0e-3, 0.15, 0.5]     # upper
    )

    # Define grids
    d_grid = np.linspace(0.6e-3, 2.4e-3, 8)
    dstar_grid = np.linspace(10e-3, 120e-3, 8)
    f_grid = np.linspace(0.02, 0.45, 8)
    snr_levels = [10.0, 20.0, 50.0, 100.0]

    theta_grid = []
    for d in d_grid:
        for dstar in dstar_grid:
            for f in f_grid:
                theta_grid.append([d, dstar, f])
    theta_grid = np.array(theta_grid)
    n_grid_points = len(theta_grid)
    n_repeats = args.n_repeats

    # Define schemes
    bvals_clinical = np.array([0, 50, 100, 200, 400, 600, 800, 1000], float)
    bvals_fit = np.array([0, 50, 200, 600, 1000], float)
    bvals_heldout = np.array([100, 400, 800], float)

    print(f"Grid size: {n_grid_points} points | Repeats: N={n_repeats}")
    print(f"Clinical sparse fit subset: {bvals_fit}")
    print(f"Clinical sparse held-out subset: {bvals_heldout}")

    # Results container
    coverage_results = []
    num_cores = os.cpu_count() or 4
    
    # --------------------------------------------------------------------------- #
    # Sweep OOD conditions
    # --------------------------------------------------------------------------- #
    conditions = {
        "baseline": {"signal_fn": biexp_signal, "noise_fn": add_rician_noise},
        "triexp": {"signal_fn": triexp_signal, "noise_fn": add_rician_noise},
        "noise_misspec": {"signal_fn": biexp_signal, "noise_fn": add_gaussian_noise}
    }

    nominal_levels = [0.50, 0.68, 0.80, 0.90, 0.95]

    for cond_name, cond in conditions.items():
        print(f"\nEvaluating OOD condition: '{cond_name}'...")
        signal_fn = cond["signal_fn"]
        noise_fn = cond["noise_fn"]

        # Track coverages for this condition
        npe_coverages = {lvl: [] for lvl in nominal_levels}
        nlls_coverages = {lvl: [] for lvl in nominal_levels}

        for snr in snr_levels:
            t0 = time.perf_counter()
            # Generate signals
            clean_full = np.array([signal_fn(bvals_clinical, *theta) for theta in theta_grid]) # (512, 8)
            tiled_clean = np.repeat(clean_full[:, None, :], n_repeats, axis=1) # (512, N, 8)
            flat_clean = tiled_clean.reshape(-1, len(bvals_clinical)) # (512*N, 8)
            
            # Add noise
            flat_noisy = noise_fn(flat_clean, snr, S0=1.0, rng=rng) # (512*N, 8)
            
            # Split into fit and held-out
            mask_fit = np.isin(bvals_clinical, bvals_fit)
            mask_heldout = np.isin(bvals_clinical, bvals_heldout)
            
            flat_noisy_fit = flat_noisy[:, mask_fit] # (512*N, 5)
            flat_noisy_heldout = flat_noisy[:, mask_heldout] # (512*N, 3)

            # Fit NLLS
            task_args = [(bvals_fit, flat_noisy_fit[i]) for i in range(len(flat_noisy_fit))]
            with multiprocessing.Pool(processes=num_cores) as pool:
                nlls_fits = pool.map(_nlls_worker, task_args)
            nlls_fits = np.array(nlls_fits) # (512*N, 3)

            # Sample NLLS posterior
            print(f"  Sampling NLLS asymptotic covariances at SNR = {snr:.1f}...")
            nlls_samples = np.zeros((args.n_noisy_samples, len(flat_noisy_fit), 3))
            for i in range(len(flat_noisy_fit)):
                nlls_samples[:, i, :] = sample_nlls_asymptotic(
                    nlls_fits[i], bvals_fit, snr, num_samples=args.n_noisy_samples, bounds=prior_bounds, rng=rng
                )

            # Sample NPE posterior
            print(f"  Sampling NPE posteriors at SNR = {snr:.1f}...")
            # format observations as sets
            bvals_fit_tiled = np.tile(bvals_fit[None, :], (len(flat_noisy_fit), 1))
            obs_torch = torch.as_tensor(np.stack([bvals_fit_tiled, flat_noisy_fit], axis=-1), dtype=torch.float32)
            snr_ctx_torch = torch.full((len(flat_noisy_fit), 1), np.log10(snr), dtype=torch.float32)
            x_torch = pack_x(obs_torch, snr_ctx_torch, "set")

            # Override condition shape and slice Standardization layer for fit subset size
            set_npe_condition_size(len(bvals_fit))

            chunk_size = 5000
            samples_npe_list = []
            for start_idx in range(0, len(x_torch), chunk_size):
                end_idx = min(start_idx + chunk_size, len(x_torch))
                with torch.no_grad():
                    chunk_samples = posterior.sample_batched((args.n_noisy_samples,), x=x_torch[start_idx:end_idx], reject_outside_prior=False)
                samples_npe_list.append(chunk_samples)
            samples_npe = torch.cat(samples_npe_list, dim=1) # (n_noisy_samples, 512*N, 3)
            samples_npe_abs = invert_theta(samples_npe, log_dstar=log_dstar).cpu().numpy() # (n_noisy_samples, 512*N, 3)
            # Clamp samples to prior bounds
            samples_npe_abs = np.clip(samples_npe_abs, prior_bounds[0], prior_bounds[1])

            # Push samples through forward biexponential model and add noise for held-out bvalues
            print(f"  Computing posterior predictive intervals at SNR = {snr:.1f}...")
            n_tot = len(flat_noisy_fit)
            pred_signals_npe = np.zeros((args.n_noisy_samples, n_tot, len(bvals_heldout)))
            pred_signals_nlls = np.zeros((args.n_noisy_samples, n_tot, len(bvals_heldout)))

            for s in range(args.n_noisy_samples):
                # NPE
                npe_theta_s = samples_npe_abs[s] # (n_tot, 3)
                clean_pred_npe = biexp_signal(bvals_heldout[None, :], npe_theta_s[:, 0:1], npe_theta_s[:, 1:2], npe_theta_s[:, 2:3]) # (n_tot, 3)
                pred_signals_npe[s] = noise_fn(clean_pred_npe, snr, S0=1.0, rng=rng)
                
                # NLLS
                nlls_theta_s = nlls_samples[s] # (n_tot, 3)
                clean_pred_nlls = biexp_signal(bvals_heldout[None, :], nlls_theta_s[:, 0:1], nlls_theta_s[:, 1:2], nlls_theta_s[:, 2:3]) # (n_tot, 3)
                pred_signals_nlls[s] = noise_fn(clean_pred_nlls, snr, S0=1.0, rng=rng)

            # Compute coverage at each nominal level
            for lvl in nominal_levels:
                alpha = 1.0 - lvl
                q_low = alpha / 2.0
                q_high = 1.0 - alpha / 2.0
                
                # NPE quantiles
                low_npe = np.percentile(pred_signals_npe, q_low * 100, axis=0) # (n_tot, 3)
                high_npe = np.percentile(pred_signals_npe, q_high * 100, axis=0) # (n_tot, 3)
                inside_npe = (flat_noisy_heldout >= low_npe) & (flat_noisy_heldout <= high_npe)
                cov_npe = inside_npe.mean() # scalar average coverage
                npe_coverages[lvl].append(cov_npe)
                
                # NLLS quantiles
                low_nlls = np.percentile(pred_signals_nlls, q_low * 100, axis=0) # (n_tot, 3)
                high_nlls = np.percentile(pred_signals_nlls, q_high * 100, axis=0) # (n_tot, 3)
                inside_nlls = (flat_noisy_heldout >= low_nlls) & (flat_noisy_heldout <= high_nlls)
                cov_nlls = inside_nlls.mean()
                nlls_coverages[lvl].append(cov_nlls)

            dt = time.perf_counter() - t0
            print(f"  SNR = {snr:.1f} completed in {dt:.2f} seconds.")

        # Average coverage over SNR levels for reporting
        for lvl in nominal_levels:
            mean_npe_cov = np.mean(npe_coverages[lvl])
            mean_nlls_cov = np.mean(nlls_coverages[lvl])
            coverage_results.append({
                "condition": cond_name,
                "nominal_level": lvl,
                "npe_coverage": mean_npe_cov,
                "nlls_coverage": mean_nlls_cov,
                "npe_deviation": mean_npe_cov - lvl,
                "nlls_deviation": mean_nlls_cov - lvl
            })

    # --------------------------------------------------------------------------- #
    # Alternative b-scheme Efficiency Sweep (tests amortization claim)
    # --------------------------------------------------------------------------- #
    print("\nEvaluating alternative b-scheme efficiency-ratio map...")
    bvals_alt = np.asarray(B_SCHEMES["optimized"], float)
    print(f"Alternative b-values: {bvals_alt}")

    alt_results = []
    for snr in snr_levels:
        t0_alt = time.perf_counter()
        print(f"  Processing alternative scheme at SNR = {snr:.1f}...")
        
        # Clean signals
        obs_clean = np.array([biexp_signal(bvals_alt, *theta) for theta in theta_grid]) # (512, 8)
        
        # NPE claimed posterior SDs
        obs_clean_tiled = np.tile(bvals_alt[None, :], (len(obs_clean), 1))
        obs_clean_torch = torch.as_tensor(np.stack([obs_clean_tiled, obs_clean], axis=-1), dtype=torch.float32)
        snr_ctx_clean_torch = torch.full((len(obs_clean), 1), np.log10(snr), dtype=torch.float32)
        x_clean = pack_x(obs_clean_torch, snr_ctx_clean_torch, "set")
        
        # Override condition shape and slice/restore Standardization layer for alt b-values
        set_npe_condition_size(len(bvals_alt))
        
        with torch.no_grad():
            samples_clean = posterior.sample_batched((1000,), x=x_clean, reject_outside_prior=False)
        samples_clean_abs = invert_theta(samples_clean, log_dstar=log_dstar)
        # Clamp samples to prior bounds
        prior_bounds_t = torch.tensor(prior_bounds, dtype=samples_clean_abs.dtype, device=samples_clean_abs.device)
        samples_clean_abs = torch.clamp(samples_clean_abs, prior_bounds_t[0], prior_bounds_t[1])
        samples_clean_disp = to_display(samples_clean_abs).numpy() # (1000, 512, 3)
        npe_post_sd = np.std(samples_clean_disp, axis=0) # (512, 3)

        # Draw repeats
        flat_clean = np.repeat(obs_clean[:, None, :], n_repeats, axis=1).reshape(-1, len(bvals_alt))
        flat_noisy = add_rician_noise(flat_clean, snr, S0=1.0, rng=rng)

        # Fit NLLS
        task_args = [(bvals_alt, flat_noisy[i]) for i in range(len(flat_noisy))]
        with multiprocessing.Pool(processes=num_cores) as pool:
            nlls_fits = pool.map(_nlls_worker, task_args)
        nlls_fits = np.array(nlls_fits).reshape(n_grid_points, n_repeats, 3)
        nlls_fits_disp = nlls_fits * np.array([1000.0, 1000.0, 1.0])
        nlls_sd = np.nanstd(nlls_fits_disp, axis=1) # (512, 3)

        # Sample NPE on noisy realizations
        bvals_tiled = np.tile(bvals_alt[None, :], (len(flat_noisy), 1))
        obs_torch = torch.as_tensor(np.stack([bvals_tiled, flat_noisy], axis=-1), dtype=torch.float32)
        snr_ctx_torch = torch.full((len(flat_noisy), 1), np.log10(snr), dtype=torch.float32)
        x_torch = pack_x(obs_torch, snr_ctx_torch, "set")

        # Override condition shape and slice/restore Standardization layer for alt b-values
        set_npe_condition_size(len(bvals_alt))

        samples_npe_list = []
        for start_idx in range(0, len(x_torch), chunk_size):
            end_idx = min(start_idx + chunk_size, len(x_torch))
            with torch.no_grad():
                chunk_samples = posterior.sample_batched((args.n_noisy_samples,), x=x_torch[start_idx:end_idx], reject_outside_prior=False)
            samples_npe_list.append(chunk_samples)
        samples_npe = torch.cat(samples_npe_list, dim=1) # (n_noisy_samples, 512*N, 3)
        samples_npe_abs = invert_theta(samples_npe, log_dstar=log_dstar)
        # Clamp samples to prior bounds
        prior_bounds_t = torch.tensor(prior_bounds, dtype=samples_npe_abs.dtype, device=samples_npe_abs.device)
        samples_npe_abs = torch.clamp(samples_npe_abs, prior_bounds_t[0], prior_bounds_t[1])
        samples_npe_disp = to_display(samples_npe_abs).numpy()
        
        npe_means = np.mean(samples_npe_disp, axis=0) # (512*N, 3)
        npe_means = npe_means.reshape(n_grid_points, n_repeats, 3)
        npe_emp_sd = np.std(npe_means, axis=1) # (512, 3)

        # Compute CRLB, ratios, and classify
        for j in range(n_grid_points):
            theta_j = theta_grid[j]
            crlb_abs = compute_crlb(theta_j, bvals_alt, snr)
            crlb_disp = crlb_abs * np.array([1000.0, 1000.0, 1.0])

            for p_idx, p_name in enumerate(["D", "Dstar", "f"]):
                c_sd = crlb_disp[p_idx]
                post_sd_j = npe_post_sd[j, p_idx]
                emp_sd_j = npe_emp_sd[j, p_idx]
                nlls_sd_j = nlls_sd[j, p_idx]

                if np.isnan(c_sd) or c_sd <= 0:
                    npe_post_ratio = np.nan
                    npe_emp_ratio = np.nan
                    nlls_ratio = np.nan
                    regime = "unknown"
                else:
                    npe_post_ratio = post_sd_j / c_sd
                    npe_emp_ratio = emp_sd_j / c_sd
                    nlls_ratio = nlls_sd_j / c_sd
                    
                    if npe_post_ratio < 0.9:
                        regime = "overconfident"
                    elif npe_post_ratio <= 1.5:
                        regime = "efficient"
                    else:
                        regime = "inefficient"

                alt_results.append({
                    "snr": snr,
                    "parameter": p_name,
                    "npe_post_ratio": npe_post_ratio,
                    "npe_emp_ratio": npe_emp_ratio,
                    "nlls_ratio": nlls_ratio,
                    "regime": regime
                })
        
        dt_alt = time.perf_counter() - t0_alt
        print(f"  SNR = {snr:.1f} completed in {dt_alt:.2f} seconds.")

    # --------------------------------------------------------------------------- #
    # Save Outputs & Generate Degradation Report
    # --------------------------------------------------------------------------- #
    model_dir = os.path.dirname(model_path) or "."
    csv_path = os.path.join(model_dir, "f1_misspecification.csv")
    print(f"\nSaving Checkpoint F1 results to {csv_path}...")

    with open(csv_path, "w", newline="") as f:
        f.write("# Checkpoint F1: Misspecification Robustness Results\n")
        f.write("# Part A: Held-out b-value posterior-predictive coverage\n")
        writer = csv.DictWriter(f, fieldnames=["condition", "nominal_level", "npe_coverage", "nlls_coverage", "npe_deviation", "nlls_deviation"])
        writer.writeheader()
        writer.writerows(coverage_results)
        
        f.write("\n# Part B: Alternative b-scheme regime counts\n")
        # Compute regime frequencies
        regime_counts = {p: {"overconfident": 0, "efficient": 0, "inefficient": 0} for p in ["D", "Dstar", "f"]}
        for row in alt_results:
            p = row["parameter"]
            reg = row["regime"]
            if reg in regime_counts[p]:
                regime_counts[p][reg] += 1
                
        f.write("parameter,overconfident_pct,efficient_pct,inefficient_pct\n")
        for p in ["D", "Dstar", "f"]:
            tot = sum(regime_counts[p].values())
            o_pct = regime_counts[p]["overconfident"] / tot * 100 if tot else 0
            e_pct = regime_counts[p]["efficient"] / tot * 100 if tot else 0
            i_pct = regime_counts[p]["inefficient"] / tot * 100 if tot else 0
            f.write(f"{p},{o_pct:.2f},{e_pct:.2f},{i_pct:.2f}\n")

    # Load baseline regime counts from efficiency_map.csv if it exists for comparison
    baseline_regimes = {}
    base_csv_path = os.path.join(model_dir, "efficiency_map.csv")
    if os.path.exists(base_csv_path):
        try:
            with open(base_csv_path, "r") as f:
                reader = csv.DictReader(row for row in f if not row.startswith("#"))
                reg_counts_base = {p: {"overconfident": 0, "efficient": 0, "inefficient": 0} for p in ["D", "Dstar", "f"]}
                for row in reader:
                    p = row["parameter"]
                    reg = row["regime"]
                    if reg in reg_counts_base[p]:
                        reg_counts_base[p][reg] += 1
                for p in ["D", "Dstar", "f"]:
                    tot = sum(reg_counts_base[p].values())
                    baseline_regimes[p] = {
                        "overconfident": reg_counts_base[p]["overconfident"] / tot * 100,
                        "efficient": reg_counts_base[p]["efficient"] / tot * 100,
                        "inefficient": reg_counts_base[p]["inefficient"] / tot * 100,
                    }
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("HELD-OUT B-VALUE COVERAGE REPORT")
    print("=" * 80)
    print(f"{'Condition':<15} | {'Nominal':<7} | {'NPE Cov':<8} {'NPE Dev':<8} | {'NLLS Cov':<8} {'NLLS Dev':<8}")
    print("-" * 80)
    for res in coverage_results:
        print(f"{res['condition']:<15} | {res['nominal_level']:<7.2f} | "
              f"{res['npe_coverage']:<8.4f} {res['npe_deviation']:<+8.4f} | "
              f"{res['nlls_coverage']:<8.4f} {res['nlls_deviation']:<+8.4f}")
    print("=" * 80)

    # --------------------------------------------------------------------------- #
    # Plotting Figures
    # --------------------------------------------------------------------------- #
    png_path = os.path.join(model_dir, "f1_misspecification.png")
    print(f"\nGenerating Checkpoint F1 summary plot to {png_path}...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left Panel: Coverage curves
    ax_cov = axes[0]
    styles = {"baseline": "-", "triexp": "--", "noise_misspec": ":"}
    colors = {"NPE": "dodgerblue", "NLLS": "forestgreen"}

    for cond_name in ["baseline", "triexp", "noise_misspec"]:
        cond_data = [r for r in coverage_results if r["condition"] == cond_name]
        nominals = [r["nominal_level"] for r in cond_data]
        npe_covs = [r["npe_coverage"] for r in cond_data]
        nlls_covs = [r["nlls_coverage"] for r in cond_data]
        
        # Sort by nominal level
        idxs = np.argsort(nominals)
        nominals = np.array(nominals)[idxs]
        npe_covs = np.array(npe_covs)[idxs]
        nlls_covs = np.array(nlls_covs)[idxs]

        ax_cov.plot(nominals, npe_covs, color=colors["NPE"], ls=styles[cond_name], marker="o", label=f"NPE ({cond_name})")
        ax_cov.plot(nominals, nlls_covs, color=colors["NLLS"], ls=styles[cond_name], marker="s", label=f"NLLS ({cond_name})")

    ax_cov.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
    ax_cov.set_xlim([0.45, 1.0])
    ax_cov.set_ylim([0.2, 1.0])
    ax_cov.set_xlabel("Nominal Credibility Level", fontsize=12)
    ax_cov.set_ylabel("Empirical Predictive Coverage", fontsize=12)
    ax_cov.set_title("Posterior-Predictive Held-out-b Coverage", fontsize=14, fontweight="bold")
    ax_cov.legend(loc="lower right")
    ax_cov.grid(True, alpha=0.3)

    # Right Panel: Regime distribution comparison
    ax_reg = axes[1]
    params = ["D", "Dstar", "f"]
    x_indices = np.arange(len(params))
    width = 0.2

    # Draw bars for alternative scheme
    alt_o = [regime_counts[p]["overconfident"] / sum(regime_counts[p].values()) * 100 for p in params]
    alt_e = [regime_counts[p]["efficient"] / sum(regime_counts[p].values()) * 100 for p in params]
    alt_i = [regime_counts[p]["inefficient"] / sum(regime_counts[p].values()) * 100 for p in params]

    # If baseline is available, draw it alongside
    if baseline_regimes:
        base_o = [baseline_regimes[p]["overconfident"] for p in params]
        base_e = [baseline_regimes[p]["efficient"] for p in params]
        base_i = [baseline_regimes[p]["inefficient"] for p in params]

        # Overconfident comparison
        ax_reg.bar(x_indices - 1.5*width, base_o, width, color="skyblue", edgecolor="k", label="Baseline (Clinical) Overconfident")
        ax_reg.bar(x_indices - 0.5*width, alt_o, width, color="blue", edgecolor="k", label="Alt (Optimized) Overconfident")
        
        # Efficient comparison
        ax_reg.bar(x_indices + 0.5*width, base_e, width, color="lightgreen", edgecolor="k", label="Baseline (Clinical) Efficient")
        ax_reg.bar(x_indices + 1.5*width, alt_e, width, color="green", edgecolor="k", label="Alt (Optimized) Efficient")
    else:
        # Just show alternative scheme
        ax_reg.bar(x_indices - width, alt_o, 2*width/3, color="dodgerblue", edgecolor="k", label="Overconfident")
        ax_reg.bar(x_indices, alt_e, 2*width/3, color="forestgreen", edgecolor="k", label="Efficient")
        ax_reg.bar(x_indices + width, alt_i, 2*width/3, color="crimson", edgecolor="k", label="Inefficient")

    ax_reg.set_xticks(x_indices)
    ax_reg.set_xticklabels(params, fontsize=12)
    ax_reg.set_ylabel("Fraction of Grid Points (%)", fontsize=12)
    ax_reg.set_title("NPE Regime Distribution: Baseline vs Alternative b-scheme", fontsize=14, fontweight="bold")
    ax_reg.legend(loc="upper right", fontsize=9)
    ax_reg.set_ylim([0, 105])
    ax_reg.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Checkpoint F1: Misspecification Robustness Summary", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {png_path}")
    print("Checkpoint F1 completed successfully!")


if __name__ == "__main__":
    main()
