"""
run_f_realdata.py
=================
Phase F: Real-Data Overconfidence Demo (Checkpoint F2).
On the brain dataset, fits NPE and NLLS on a subset of b-values, predicts
held-out b-values, and computes posterior-predictive coverage and normalized residuals.
"""
from __future__ import annotations

import os
import sys
import time
import csv
import argparse
import numpy as np
import scipy.optimize as opt
import torch
import nibabel as nib

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Handle imports robustly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_prior import get_processed_prior, DISPLAY_UNITS, DISPLAY_SCALE, to_display, invert_theta
from train_npe import pack_x, SNRWrapperEmbedding


# --------------------------------------------------------------------------- #
# Fit helpers (copied from validation/efficiency scripts for safety)
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


def biexp_signal(bvals, D, Dstar, f, S0=1.0):
    """Clean biexponential signal."""
    bvals = np.asarray(bvals, dtype=float)
    return S0 * (f * np.exp(-bvals * Dstar) + (1.0 - f) * np.exp(-bvals * D))


def add_rician_noise(signal, snr, S0=1.0, rng=None):
    """Standard Rician noise."""
    rng = np.random.default_rng() if rng is None else rng
    sigma = S0 / snr
    signal = np.asarray(signal, dtype=float)
    n_real = rng.normal(0.0, sigma, size=signal.shape)
    n_imag = rng.normal(0.0, sigma, size=signal.shape)
    return np.sqrt((signal + n_real) ** 2 + n_imag ** 2)


def sample_nlls_asymptotic(theta_fit, bvals_fit, snr, num_samples=200, bounds=None, rng=None):
    """
    Draws parameter samples for NLLS using the Fisher Information Matrix covariance.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    D, Dstar, f = theta_fit[0], theta_fit[1], theta_fit[2]
    if np.isnan(D) or np.isnan(Dstar) or np.isnan(f):
        if bounds is not None:
            return rng.uniform(bounds[0], bounds[1], size=(num_samples, 3))
        return np.zeros((num_samples, 3))
    
    J = compute_jacobian(theta_fit, bvals_fit, S0=1.0)
    sigma = 1.0 / snr
    FIM = (J.T @ J) / (sigma ** 2)
    try:
        cov = np.linalg.inv(FIM)
        cov += 1e-12 * np.eye(3)
        samples = rng.multivariate_normal(theta_fit, cov, size=num_samples)
    except (np.linalg.LinAlgError, ValueError):
        diag_var = np.array([0.5e-3, 20e-3, 0.1]) ** 2
        cov = np.diag(diag_var) / (snr ** 2)
        samples = rng.multivariate_normal(theta_fit, cov, size=num_samples)
        
    if bounds is not None:
        samples = np.clip(samples, bounds[0], bounds[1])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Checkpoint F2 Real-Data Demo.")
    parser.add_argument("--model", type=str, default="npe/npe_posterior_setB.pt", help="Path to model file.")
    parser.add_argument("--suffix", type=str, default="_f2", help="Suffix for output files.")
    parser.add_argument("--n-voxels", type=int, default=500, help="Number of gray matter voxels to evaluate.")
    parser.add_argument("--n-noisy-samples", type=int, default=200, help="Number of posterior samples to draw.")
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("Checkpoint F2: Real-Data Overconfidence Demo")
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

    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    print(f"Auto-detected log_dstar = {log_dstar}")

    prior, _, _ = get_processed_prior(device="cpu", log_dstar=log_dstar)
    prior_bounds = (
        [0.2e-3, 3.0e-3, 0.0],  # lower
        [3.0e-3, 0.15, 0.5]     # upper
    )

    # Save original standardize mean and std to support dynamic set size adjustments
    estimator = posterior.posterior_estimator
    std_mod = estimator.net._embedding_net[0]
    orig_mean = std_mod._mean.clone()
    orig_std = std_mod._std.clone()

    def set_npe_condition_size(K: int):
        estimator._condition_shape = torch.Size([K, 3])
        std_mod._mean = orig_mean[:K, :]
        std_mod._std = orig_std[:K, :]

    # 2. Load and Preprocess Real Data
    img_path = "download/Data/brain.nii.gz"
    mask_path = "download/Data/brain_mask_gray_matter.nii.gz"
    bval_path = "download/Data/brain.bval"

    if not os.path.exists(img_path) or not os.path.exists(mask_path) or not os.path.exists(bval_path):
        print("Error: Brain dataset not found in download/Data/. Skipping F2 real-data demo.")
        return

    print("Loading brain dataset and gray matter mask...")
    img = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    bvals_raw = np.genfromtxt(bval_path, dtype=float)

    # Identify unique b-values and their repetitions
    unique_bvals, counts = np.unique(bvals_raw, return_counts=True)
    print("Unique b-values in data:", unique_bvals)

    # Partition b-values into fit and held-out (matching clinical sparse scheme)
    bvals_fit = np.array([0, 50, 200, 600, 1000], float)
    bvals_heldout = np.array([100, 400, 800], float)
    print("Subset bvals for fitting:", bvals_fit)
    print("Subset bvals held-out:", bvals_heldout)

    # Find active voxels
    coords = np.argwhere(mask > 0)
    print(f"Total gray matter voxels: {len(coords)}")

    # Randomly select a subset of voxels
    sample_indices = rng.choice(len(coords), size=args.n_voxels, replace=False)
    selected_coords = coords[sample_indices]

    # Containers for voxel data
    fit_signals = []
    heldout_signals = []
    estimated_snrs = []

    print(f"Preprocessing {args.n_voxels} selected voxels (trace-averaging and estimating SNR)...")
    for coord in selected_coords:
        voxel_signal = img[coord[0], coord[1], coord[2], :]
        
        # 1. Trace-average over directions
        avg_signal = []
        for val in unique_bvals:
            avg_signal.append(np.mean(voxel_signal[bvals_raw == val]))
        avg_signal = np.array(avg_signal)
        
        # Normalize by S(0)
        s0 = avg_signal[0]
        norm_signal = avg_signal / s0 if s0 > 0 else avg_signal
        
        # 2. Estimate noise std dev from repetitions
        vars_rep = []
        for val in unique_bvals:
            if val == 0:
                continue
            reps = voxel_signal[bvals_raw == val]
            vars_rep.append(np.var(reps, ddof=1))
        
        sigma_est = np.sqrt(np.mean(vars_rep))
        snr = s0 / sigma_est if (sigma_est > 0 and s0 > 0) else 30.0
        snr = np.clip(snr, 8.0, 100.0) # Clamp to prior range
        
        # Separate into fit and held-out
        fit_sig = [norm_signal[np.where(unique_bvals == b)[0][0]] for b in bvals_fit]
        heldout_sig = [norm_signal[np.where(unique_bvals == b)[0][0]] for b in bvals_heldout]
        
        fit_signals.append(fit_sig)
        heldout_signals.append(heldout_sig)
        estimated_snrs.append(snr)

    fit_signals = np.array(fit_signals) # (n_voxels, 5)
    heldout_signals = np.array(heldout_signals) # (n_voxels, 3)
    estimated_snrs = np.array(estimated_snrs) # (n_voxels,)

    # 3. Fit NLLS on Fit Subset
    print("Fitting NLLS on fit subset...")
    nlls_fits = []
    for i in range(args.n_voxels):
        nlls_fits.append(fit_biexp_nlls(bvals_fit, fit_signals[i]))
    nlls_fits = np.array(nlls_fits) # (n_voxels, 3)

    # 4. Sample NLLS Posterior
    print("Sampling NLLS asymptotic covariances...")
    nlls_samples = np.zeros((args.n_noisy_samples, args.n_voxels, 3))
    for i in range(args.n_voxels):
        nlls_samples[:, i, :] = sample_nlls_asymptotic(
            nlls_fits[i], bvals_fit, estimated_snrs[i], num_samples=args.n_noisy_samples, bounds=prior_bounds, rng=rng
        )

    # 5. Sample NPE Posterior
    print("Sampling NPE posteriors...")
    bvals_fit_tiled = np.tile(bvals_fit[None, :], (args.n_voxels, 1))
    obs_torch = torch.as_tensor(np.stack([bvals_fit_tiled, fit_signals], axis=-1), dtype=torch.float32)
    snr_ctx_torch = torch.as_tensor(np.log10(estimated_snrs)[:, None], dtype=torch.float32)
    x_torch = pack_x(obs_torch, snr_ctx_torch, "set")

    # Override shape and slice standardize layer
    set_npe_condition_size(len(bvals_fit))

    with torch.no_grad():
        samples_npe = posterior.sample_batched((args.n_noisy_samples,), x=x_torch, reject_outside_prior=False) # (n_noisy_samples, n_voxels, 3)
    samples_npe_abs = invert_theta(samples_npe, log_dstar=log_dstar).cpu().numpy()
    # Clamp samples to prior bounds
    samples_npe_abs = np.clip(samples_npe_abs, prior_bounds[0], prior_bounds[1])

    # 6. Predict Held-out Signals (adding trace-average noise: SNR_avg = SNR * sqrt(6))
    print("Generating posterior predictive samples for held-out b-values...")
    pred_signals_npe = np.zeros((args.n_noisy_samples, args.n_voxels, len(bvals_heldout)))
    pred_signals_nlls = np.zeros((args.n_noisy_samples, args.n_voxels, len(bvals_heldout)))

    for i in range(args.n_voxels):
        snr_i = estimated_snrs[i]
        # Average SNR is scaled by sqrt(6) because we averaged 6 directions
        snr_avg = snr_i * np.sqrt(6.0)
        
        for s in range(args.n_noisy_samples):
            # NPE
            theta_npe = samples_npe_abs[s, i]
            clean_npe = biexp_signal(bvals_heldout, *theta_npe)
            pred_signals_npe[s, i] = add_rician_noise(clean_npe, snr_avg, S0=1.0, rng=rng)
            
            # NLLS
            theta_nlls = nlls_samples[s, i]
            clean_nlls = biexp_signal(bvals_heldout, *theta_nlls)
            pred_signals_nlls[s, i] = add_rician_noise(clean_nlls, snr_avg, S0=1.0, rng=rng)

    # 7. Compute Coverage
    print("Computing posterior-predictive coverage...")
    nominal_levels = [0.50, 0.68, 0.80, 0.90, 0.95]
    coverage_rows = []

    for lvl in nominal_levels:
        alpha = 1.0 - lvl
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0
        
        # NPE
        low_npe = np.percentile(pred_signals_npe, q_low * 100, axis=0) # (n_voxels, 3)
        high_npe = np.percentile(pred_signals_npe, q_high * 100, axis=0)
        inside_npe = (heldout_signals >= low_npe) & (heldout_signals <= high_npe)
        cov_npe = inside_npe.mean()
        
        # NLLS
        low_nlls = np.percentile(pred_signals_nlls, q_low * 100, axis=0)
        high_nlls = np.percentile(pred_signals_nlls, q_high * 100, axis=0)
        inside_nlls = (heldout_signals >= low_nlls) & (heldout_signals <= high_nlls)
        cov_nlls = inside_nlls.mean()

        coverage_rows.append({
            "nominal_level": lvl,
            "npe_coverage": cov_npe,
            "nlls_coverage": cov_nlls,
            "npe_deviation": cov_npe - lvl,
            "nlls_deviation": cov_nlls - lvl
        })

    # Save to CSV
    model_dir = os.path.dirname(model_path) or "."
    csv_path = os.path.join(model_dir, "f2_realdata.csv")
    print(f"Saving real-data results to {csv_path}...")
    with open(csv_path, "w", newline="") as f:
        f.write("# Checkpoint F2: Real-Data Overconfidence Demo Results\n")
        writer = csv.DictWriter(f, fieldnames=["nominal_level", "npe_coverage", "nlls_coverage", "npe_deviation", "nlls_deviation"])
        writer.writeheader()
        writer.writerows(coverage_rows)

    # 8. Compute Normalized Residuals (z-scores) on held-out b-values
    # z = (y_heldout - mean(y_pred)) / std(y_pred)
    mean_pred_npe = np.mean(pred_signals_npe, axis=0) # (n_voxels, 3)
    std_pred_npe = np.std(pred_signals_npe, axis=0)
    residuals_npe = (heldout_signals - mean_pred_npe) / std_pred_npe

    mean_pred_nlls = np.mean(pred_signals_nlls, axis=0)
    std_pred_nlls = np.std(pred_signals_nlls, axis=0)
    residuals_nlls = (heldout_signals - mean_pred_nlls) / std_pred_nlls

    # Flatten residuals across voxels and held-out b-values
    residuals_npe_flat = residuals_npe.ravel()
    residuals_nlls_flat = residuals_nlls.ravel()

    # Print summary
    print("\n" + "=" * 80)
    print("REAL-DATA HELD-OUT B-VALUE COVERAGE REPORT")
    print("=" * 80)
    print(f"{'Nominal':<7} | {'NPE Cov':<8} {'NPE Dev':<8} | {'NLLS Cov':<8} {'NLLS Dev':<8}")
    print("-" * 80)
    for res in coverage_rows:
        print(f"{res['nominal_level']:<7.2f} | "
              f"{res['npe_coverage']:<8.4f} {res['npe_deviation']:<+8.4f} | "
              f"{res['nlls_coverage']:<8.4f} {res['nlls_deviation']:<+8.4f}")
    print("=" * 80)

    # Plot figure
    png_path = os.path.join(model_dir, "f2_realdata.png")
    print(f"\nGenerating Checkpoint F2 summary plot to {png_path}...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left Panel: Coverage curve
    ax_cov = axes[0]
    nominals = [r["nominal_level"] for r in coverage_rows]
    npe_covs = [r["npe_coverage"] for r in coverage_rows]
    nlls_covs = [r["nlls_coverage"] for r in coverage_rows]

    ax_cov.plot(nominals, npe_covs, "bo-", label="NPE (Real Data)", lw=2)
    ax_cov.plot(nominals, nlls_covs, "gs-", label="NLLS (Real Data)", lw=2)
    ax_cov.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ideal")
    ax_cov.set_xlim([0.45, 1.0])
    ax_cov.set_ylim([0.2, 1.0])
    ax_cov.set_xlabel("Nominal Credibility Level", fontsize=12)
    ax_cov.set_ylabel("Empirical Predictive Coverage", fontsize=12)
    ax_cov.set_title("Real-Data Posterior-Predictive Coverage", fontsize=14, fontweight="bold")
    ax_cov.legend(loc="lower right")
    ax_cov.grid(True, alpha=0.3)

    # Right Panel: Residual z-score distributions
    ax_res = axes[1]
    bins = np.linspace(-6, 6, 41)
    
    # Plot standard normal for reference
    x_grid = np.linspace(-6, 6, 200)
    pdf = np.exp(-0.5 * x_grid**2) / np.sqrt(2 * np.pi)
    ax_res.plot(x_grid, pdf, "k--", label="Standard Normal N(0,1)", lw=2)

    ax_res.hist(residuals_npe_flat, bins=bins, density=True, alpha=0.5, color="blue", label="NPE Residuals", edgecolor="blue", histtype="stepfilled")
    ax_res.hist(residuals_nlls_flat, bins=bins, density=True, alpha=0.5, color="green", label="NLLS Residuals", edgecolor="green", histtype="step", lw=2)

    # Annotate standard deviations
    std_npe_res = np.std(residuals_npe_flat)
    std_nlls_res = np.std(residuals_nlls_flat)
    ax_res.text(0.05, 0.95, f"NPE Residual SD: {std_npe_res:.2f}\nNLLS Residual SD: {std_nlls_res:.2f}",
                transform=ax_res.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))

    ax_res.set_xlabel("Normalized Residual (z-score)", fontsize=12)
    ax_res.set_ylabel("Density", fontsize=12)
    ax_res.set_title("Held-out b-value Residual Distributions", fontsize=14, fontweight="bold")
    ax_res.legend(loc="upper right")
    ax_res.grid(True, alpha=0.3)

    fig.suptitle(f"Checkpoint F2: Real-Data Overconfidence Demo (N={args.n_voxels} voxels)", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {png_path}")
    print("Checkpoint F2 completed successfully!")


if __name__ == "__main__":
    main()
