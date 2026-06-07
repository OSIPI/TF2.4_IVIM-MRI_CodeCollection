"""
run_cp3_validation.py
======================
Checkpoint 3: Validation and benchmarking of the trained setB posterior.
Performs:
1. SBC ranks and check_sbc stats, stratified into 3 SNR bins.
2. Expected coverage (TARP) and direct nominal quantile-interval coverage.
3. comparison of NPE posterior SD, analytical sqrt_crlb, and NLLS empirical SD.
"""
from __future__ import annotations

import os
import time
import json
import csv
import numpy as np
import scipy.optimize as opt
import torch

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbi.diagnostics import run_sbc, check_sbc, run_tarp
from sbi.analysis import sbc_rank_plot

from npe_prior import get_processed_prior, DISPLAY_UNITS, DISPLAY_SCALE, to_display, invert_theta
from npe_simulator import IVIMNPESimulator, B_SCHEMES
from train_npe import pack_x, SNRWrapperEmbedding
from ivim_simulator import add_rician_noise


# --------------------------------------------------------------------------- #
# CRLB and Fitting Helpers
# --------------------------------------------------------------------------- #

def compute_jacobian(theta: np.ndarray | list, bvals: np.ndarray, S0: float = 1.0) -> np.ndarray:
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


def compute_crlb(theta: np.ndarray | list, bvals: np.ndarray, snr: float, S0: float = 1.0) -> np.ndarray:
    """
    Computes analytical standard deviation lower bounds (sqrt_crlb) for [D, Dstar, f] in absolute units.
    FIM = J.T @ J / sigma^2, sigma = S0 / SNR.
    CRLB = diag(inv(FIM)).
    Returns np.nan if singular.
    """
    J = compute_jacobian(theta, bvals, S0)
    sigma = S0 / snr
    
    # FIM = J.T @ J / sigma^2
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
    Performs a bounded non-linear least squares fit to biexponential model using scipy.
    Bounds match prior: D in [0.2e-3, 3.0e-3], Dstar in [3.0e-3, 0.15], f in [0.0, 0.5].
    """
    bounds = (
        [0.2e-3, 3.0e-3, 0.0],  # lower
        [3.0e-3, 0.15, 0.5]     # upper
    )
    # Start in middle of prior box
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


# --------------------------------------------------------------------------- #
# Validation runner
# --------------------------------------------------------------------------- #

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run CP3 validation.")
    parser.add_argument("--model", type=str, default="npe/npe_posterior_setB.pt", help="Path to model file.")
    parser.add_argument("--suffix", type=str, default="_v2", help="Suffix for output files.")
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("=" * 80)
    print("Phase E - Checkpoint 3: Calibration Validation")
    print("=" * 80)

    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback if run from repo root or vice versa
        alt_path = "npe/" + os.path.basename(model_path) if not model_path.startswith("npe/") else os.path.basename(model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find model at {model_path}")

    # Resolve pickling namespace dynamically
    import sys
    sys.modules['__main__'].SNRWrapperEmbedding = SNRWrapperEmbedding

    print(f"Loading model from {model_path}...")
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)

    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    print(f"Auto-detected log_dstar = {log_dstar}")

    prior, n_params, _ = get_processed_prior(device="cpu", log_dstar=log_dstar)
    sim = IVIMNPESimulator(representation="set", clean=False, seed=seed)
    active_bvals = sim.active_bvals

    model_dir = os.path.dirname(model_path)
    if not model_dir:
        model_dir = "."

    # -- 1. Generate Validation Dataset (1000 samples) --
    n_val = 1000
    print(f"Generating {n_val} validation samples...")
    thetas_val = prior.sample((n_val,))
    snrs_val = sim.sample_snr(n_val, rng=np.random.default_rng(seed))
    # Convert validation parameters to absolute units for simulator
    thetas_val_abs = invert_theta(thetas_val, log_dstar=log_dstar)
    obs_val, snr_ctx_val = sim(thetas_val_abs, snr=snrs_val)
    x_val = pack_x(obs_val, snr_ctx_val, "set")

    # Define bins
    # SNR low: [8.0, 18.57), mid: [18.57, 43.11), high: [43.11, 100.0]
    mask_low = (snrs_val >= 8.0) & (snrs_val < 18.57)
    mask_mid = (snrs_val >= 18.57) & (snrs_val < 43.11)
    mask_high = (snrs_val >= 43.11) & (snrs_val <= 100.0)

    masks = {
        "low": mask_low,
        "mid": mask_mid,
        "high": mask_high
    }
    
    for k, m in masks.items():
        print(f"  SNR bin '{k}': {int(m.sum())} samples (range: [{snrs_val[m].min():.2f}, {snrs_val[m].max():.2f}])")

    # -- 2. Run SBC (Simulation-Based Calibration) --
    print("\nRunning SBC ranking...")
    num_post_samples = 1000
    ranks, joint_ranks = run_sbc(
        thetas_val,
        x_val,
        posterior,
        num_posterior_samples=num_post_samples,
        show_progress_bar=True
    )

    # Stratify and check SBC
    sbc_stats = []
    print("\nEvaluating SBC uniformity metrics...")
    
    # Save rank histograms faceted by SNR bin
    fig_sbc, axes_sbc = plt.subplots(3, 3, figsize=(15, 12))
    
    bin_names = ["low", "mid", "high"]
    for row_idx, bin_name in enumerate(bin_names):
        mask = masks[bin_name]
        ranks_bin = ranks[mask]
        thetas_bin = thetas_val[mask]
        x_bin = x_val[mask]
        
        # Draw DAP samples (1 sample per validation run)
        with torch.no_grad():
            dap_samples_bin = posterior.sample_batched((1,), x=x_bin).squeeze(0)
            
        check = check_sbc(
            ranks_bin,
            thetas_bin,
            dap_samples_bin,
            num_posterior_samples=num_post_samples
        )
        
        ks_pvals = check["ks_pvals"].numpy()
        c2st_ranks = check["c2st_ranks"].numpy()
        c2st_dap = check["c2st_dap"].numpy()
        
        print(f"SNR Bin: {bin_name.upper()}")
        print(f"  KS p-values (D, Dstar, f): {np.round(ks_pvals, 4)}")
        print(f"  C2ST Ranks accuracy:     {np.round(c2st_ranks, 4)} (0.5 = perfectly uniform)")
        print(f"  C2ST DAP accuracy:       {np.round(c2st_dap, 4)}")
        
        for i, name in enumerate(["D", "Dstar", "f"]):
            sbc_stats.append({
                "snr_bin": bin_name,
                "parameter": name,
                "ks_pval": ks_pvals[i],
                "c2st_ranks": c2st_ranks[i],
                "c2st_dap": c2st_dap[i]
            })
            
        # Plot ranks for this row
        sbc_rank_plot(
            ranks_bin,
            num_posterior_samples=num_post_samples,
            ax=axes_sbc[row_idx],
            parameter_labels=DISPLAY_UNITS,
            plot_type="hist"
        )
        
        # Add labels to y axis of first column to mark SNR bins
        axes_sbc[row_idx, 0].set_ylabel(f"{bin_name.upper()} SNR\n\nDensity", fontsize=12)

    # Save SBC stats to CSV
    csv_sbc_path = os.path.join(model_dir, f"cp3_sbc_stats{args.suffix}.csv")
    with open(csv_sbc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_bin", "parameter", "ks_pval", "c2st_ranks", "c2st_dap"])
        writer.writeheader()
        writer.writerows(sbc_stats)
    print(f"SBC uniformity statistics saved to {csv_sbc_path}")

    # Finalize and save SBC plot
    fig_sbc.suptitle("Simulation-Based Calibration (SBC) Rank Histograms per SNR Bin", fontsize=16, y=0.98)
    fig_sbc.tight_layout()
    sbc_png_path = os.path.join(model_dir, f"cp3_sbc_ranks{args.suffix}.png")
    fig_sbc.savefig(sbc_png_path, dpi=300)
    plt.close(fig_sbc)
    print(f"SBC rank histograms saved to {sbc_png_path}")

    # -- 3. Coverage Analysis --
    print("\nRunning Expected Coverage checks...")
    nominal_levels = [0.50, 0.68, 0.80, 0.90, 0.95]
    coverage_rows = []
    
    # Setup plotting for TARP and direct coverage curves
    fig_cov, axes_cov = plt.subplots(1, 3, figsize=(16, 5))
    colors_bin = {"low": "blue", "mid": "green", "high": "orange"}
    
    for bin_name in bin_names:
        mask = masks[bin_name]
        thetas_bin = thetas_val[mask]
        x_bin = x_val[mask]
        M_bin = int(mask.sum())
        
        # TARP check
        ecp_tarp, alpha_tarp = run_tarp(
            thetas_bin,
            x_bin,
            posterior,
            num_posterior_samples=num_post_samples,
            show_progress_bar=False
        )
        
        # Direct Coverage
        print(f"Sampling posteriors for direct coverage (bin: {bin_name})...")
        with torch.no_grad():
            samples_bin = posterior.sample_batched((1000,), x=x_bin) # (1000, M_bin, 3)
            
        thetas_bin_abs = invert_theta(thetas_bin, log_dstar=log_dstar)
        samples_bin_abs = invert_theta(samples_bin, log_dstar=log_dstar)
            
        for lvl in nominal_levels:
            alpha = 1.0 - lvl
            q_low = alpha / 2.0
            q_high = 1.0 - alpha / 2.0
            
            # Compute credible intervals per validation run
            low_quantiles = torch.quantile(samples_bin_abs, q_low, dim=0) # (M_bin, 3)
            high_quantiles = torch.quantile(samples_bin_abs, q_high, dim=0) # (M_bin, 3)
            
            inside = (thetas_bin_abs >= low_quantiles) & (thetas_bin_abs <= high_quantiles) # (M_bin, 3)
            empirical = inside.float().mean(dim=0).numpy() # (3,)
            
            for i, name in enumerate(["D", "Dstar", "f"]):
                dev = empirical[i] - lvl
                flagged = abs(dev) > 0.05
                coverage_rows.append({
                    "snr_bin": bin_name,
                    "parameter": name,
                    "nominal_level": lvl,
                    "empirical_coverage": empirical[i],
                    "deviation": dev,
                    "flagged": flagged
                })
                
        # Plot coverage curves (Nominal vs Empirical) for each parameter
        # We will compute coverage for a dense grid of nominal levels for smooth curves
        dense_nominal = np.linspace(0.0, 1.0, 21)
        dense_empirical = np.zeros((len(dense_nominal), 3))
        
        for idx, lvl in enumerate(dense_nominal):
            if lvl == 0.0:
                dense_empirical[idx, :] = 0.0
            elif lvl == 1.0:
                dense_empirical[idx, :] = 1.0
            else:
                alpha = 1.0 - lvl
                low_q = torch.quantile(samples_bin_abs, alpha / 2.0, dim=0)
                high_q = torch.quantile(samples_bin_abs, 1.0 - alpha / 2.0, dim=0)
                inside = (thetas_bin_abs >= low_q) & (thetas_bin_abs <= high_q)
                dense_empirical[idx, :] = inside.float().mean(dim=0).numpy()
                
        for i, name in enumerate(["D", "Dstar", "f"]):
            ax = axes_cov[i]
            ax.plot(dense_nominal, dense_empirical[:, i], label=f"{bin_name.upper()} SNR", color=colors_bin[bin_name], lw=2)

    # Finalize coverage curves plot
    for i, name in enumerate(DISPLAY_UNITS):
        ax = axes_cov[i]
        ax.plot([0, 1], [0, 1], "k--", label="Ideal Reference")
        ax.set_title(f"Coverage Curve: {name}", fontsize=12)
        ax.set_xlabel("Nominal Credibility", fontsize=10)
        ax.set_ylabel("Empirical Coverage", fontsize=10)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    fig_cov.suptitle("Direct Quantile Coverage Curves", fontsize=16, y=0.98)
    fig_cov.tight_layout()
    cov_png_path = os.path.join(model_dir, f"cp3_coverage_curves{args.suffix}.png")
    fig_cov.savefig(cov_png_path, dpi=300)
    plt.close(fig_cov)
    print(f"Coverage curves plot saved to {cov_png_path}")

    # Save coverage table
    csv_cov_path = os.path.join(model_dir, f"cp3_coverage{args.suffix}.csv")
    with open(csv_cov_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_bin", "parameter", "nominal_level", "empirical_coverage", "deviation", "flagged"])
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"Coverage results table saved to {csv_cov_path}")

    # -- 4. CRLB Comparison --
    print("\nRunning CRLB + NLLS Benchmarking...")
    # Anchor truths from ivim_simulator.py (absolute units)
    anchor_truths = [
        {"name": "Parenchyma typical", "D": 1.1e-3, "Dstar": 30.0e-3, "f": 0.15},
        {"name": "High perfusion", "D": 1.3e-3, "Dstar": 50.0e-3, "f": 0.25},
        {"name": "Low-f hard", "D": 0.9e-3, "Dstar": 1.5e-2, "f": 0.10}
    ]
    snr_levels = [10.0, 20.0, 50.0, 100.0]
    n_repeats = 500
    
    crlb_compare_rows = []

    # 3x3 plot structure: Rows = Parameters (D, Dstar, f), Cols = Anchor Truths
    fig_crlb, axes_crlb = plt.subplots(3, 3, figsize=(15, 12))
    
    # We will build grids of SNR vs values to plot curves
    for col_idx, truth in enumerate(anchor_truths):
        print(f"Evaluating {truth['name']} [D={truth['D']:.2e}, D*={truth['Dstar']:.2e}, f={truth['f']:.2f}]...")
        theta_abs = np.array([truth["D"], truth["Dstar"], truth["f"]])
        
        # Display scale truth values
        theta_disp = to_display(theta_abs).numpy()
        
        # Arrays to collect values for plotting
        cr_vals = {p: [] for p in ["D", "Dstar", "f"]}
        npe_vals = {p: [] for p in ["D", "Dstar", "f"]}
        nlls_vals = {p: [] for p in ["D", "Dstar", "f"]}

        for snr in snr_levels:
            # 1. Analytical CRLB (scale diffusivities by 1000)
            crlb_abs = compute_crlb(theta_abs, active_bvals, snr)
            crlb_disp = crlb_abs * np.array([1000.0, 1000.0, 1.0])
            
            # 2. Simulate repeats
            clean_sig = sim.clean_signals(theta_abs[None, :], active_bvals).ravel()
            tiled_clean = np.broadcast_to(clean_sig, (n_repeats, len(active_bvals)))
            noisy_signals = add_rician_noise(tiled_clean, snr, S0=1.0, rng=np.random.default_rng(seed))
            
            # 3. Fit repeats using NLLS
            nlls_fits_abs = np.zeros((n_repeats, 3))
            for r in range(n_repeats):
                nlls_fits_abs[r] = fit_biexp_nlls(active_bvals, noisy_signals[r])
                
            # NLLS Empirical SD (scale diffusivities by 1000)
            # Remove NaNs before computing SD
            nlls_fits_disp = nlls_fits_abs * np.array([1000.0, 1000.0, 1.0])
            nlls_sd_disp = np.nanstd(nlls_fits_disp, axis=0)

            # 4. NPE Posterior SD
            # Format inputs for model
            obs_torch = torch.as_tensor(np.stack([np.broadcast_to(active_bvals, (n_repeats, len(active_bvals))), noisy_signals], axis=-1), dtype=torch.float32)
            snr_ctx_torch = torch.full((n_repeats, 1), np.log10(snr), dtype=torch.float32)
            x_torch = pack_x(obs_torch, snr_ctx_torch, "set")
            
            # Draw samples
            with torch.no_grad():
                # 1000 samples per repeat
                samples_npe = posterior.sample_batched((1000,), x=x_torch) # (1000, n_repeats, 3)
                
            # Convert to absolute and display scale
            samples_npe_abs = invert_theta(samples_npe, log_dstar=log_dstar)
            samples_npe_disp = to_display(samples_npe_abs).numpy() # (1000, n_repeats, 3)
            # Standard deviation per repeat, then average
            npe_sds_per_repeat = np.std(samples_npe_disp, axis=0) # (n_repeats, 3)
            npe_sd_disp = np.mean(npe_sds_per_repeat, axis=0) # (3,)
            
            for i, p in enumerate(["D", "Dstar", "f"]):
                cr_vals[p].append(crlb_disp[i])
                npe_vals[p].append(npe_sd_disp[i])
                nlls_vals[p].append(nlls_sd_disp[i])
                
                crlb_compare_rows.append({
                    "anchor_case": truth["name"],
                    "snr": snr,
                    "parameter": p,
                    "true_value": theta_disp[i],
                    "crlb_sd": crlb_disp[i],
                    "nlls_empirical_sd": nlls_sd_disp[i],
                    "npe_posterior_sd": npe_sd_disp[i]
                })

        # Plot comparison curves for this anchor truth
        for i, p in enumerate(["D", "Dstar", "f"]):
            ax = axes_crlb[i, col_idx]
            ax.plot(snr_levels, cr_vals[p], "k--", label="Analytical CRLB", lw=1.5)
            ax.plot(snr_levels, nlls_vals[p], "go:", label="NLLS Empirical SD", markersize=5, lw=1.5)
            ax.plot(snr_levels, npe_vals[p], "bs-", label="NPE Posterior SD", markersize=5, lw=1.5)
            
            ax.set_xscale("log")
            ax.set_xticks(snr_levels)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(True, which="both", alpha=0.3)
            
            if col_idx == 0:
                ax.set_ylabel(DISPLAY_UNITS[i], fontsize=12)
            if i == 0:
                ax.set_title(truth["name"], fontsize=12)
            if i == 2:
                ax.set_xlabel("SNR", fontsize=10)
            if i == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

    fig_crlb.suptitle("Efficiency Benchmark Comparison: NPE SD vs NLLS SD vs CRLB", fontsize=16, y=0.98)
    fig_crlb.tight_layout()
    crlb_png_path = os.path.join(model_dir, f"cp3_crlb_compare{args.suffix}.png")
    fig_crlb.savefig(crlb_png_path, dpi=300)
    plt.close(fig_crlb)
    print(f"CRLB compare plot saved to {crlb_png_path}")

    # Save CRLB stats to CSV
    csv_crlb_path = os.path.join(model_dir, f"cp3_crlb_compare{args.suffix}.csv")
    with open(csv_crlb_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anchor_case", "snr", "parameter", "true_value", "crlb_sd", "nlls_empirical_sd", "npe_posterior_sd"])
        writer.writeheader()
        writer.writerows(crlb_compare_rows)
    print(f"CRLB efficiency comparison table saved to {csv_crlb_path}")

    # -- 5. Generate Verdict --
    print("\n" + "=" * 80)
    print("CALIBRATION VERDICT")
    print("=" * 80)
    
    # Analyze results
    n_flagged = sum(1 for row in coverage_rows if row["flagged"])
    all_ks_pvals = [row["ks_pval"] for row in sbc_stats]
    min_ks = min(all_ks_pvals)
    mean_ks = np.mean(all_ks_pvals)
    
    print(f"Direct coverage flags (>0.05 deviation): {n_flagged} out of {len(coverage_rows)} cells.")
    print(f"SBC Uniformity KS test p-values: min={min_ks:.4e}, mean={mean_ks:.4f}")
    
    if n_flagged == 0 and min_ks > 0.01:
        print("Verdict: ACCEPTABLE (No coverage violations, ranks are uniform).")
    elif n_flagged <= 5:
        print("Verdict: ACCEPTABLE WITH MINOR DEVIATIONS (Low number of coverage violations).")
    else:
        print("Verdict: UNACCEPTABLE (High deviation from nominal coverage).")
    print("=" * 80)


if __name__ == "__main__":
    main()
