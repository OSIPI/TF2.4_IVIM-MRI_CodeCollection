"""
run_e_efficiency.py
===================
Phase E (Path B): NPE vs CRLB vs NLLS Efficiency Map.
Calculates the claimed efficiency (npe_post_ratio), true empirical efficiency (npe_emp_ratio),
and NLLS baseline efficiency (nlls_ratio) across a 2048-point parameter-SNR grid.
Generates efficiency_map.csv and efficiency_map.png.
"""
from __future__ import annotations

import os
import sys
import time
import csv
import argparse
import multiprocessing
import numpy as np
import torch

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Handle imports robustly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_prior import get_processed_prior, DISPLAY_UNITS, DISPLAY_SCALE, to_display, invert_theta, PRIOR_LOW, PRIOR_HIGH
from npe_simulator import IVIMNPESimulator
from train_npe import pack_x, SNRWrapperEmbedding
from ivim_simulator import add_rician_noise, B_SCHEMES
from run_cp3_validation import compute_crlb, fit_biexp_nlls


def _nlls_worker(args):
    bvals, signal = args
    return fit_biexp_nlls(bvals, signal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase E Efficiency Map.")
    parser.add_argument("--model", type=str, default="npe/npe_posterior_setB.pt", help="Path to model file.")
    parser.add_argument("--suffix", type=str, default="_v2", help="Suffix for output files.")
    parser.add_argument("--n-repeats", type=int, default=50, help="Number of noise realizations (N).")
    parser.add_argument("--n-noisy-samples", type=int, default=200, help="Number of posterior samples to draw per noisy realization.")
    parser.add_argument("--out-tag", type=str, default="", help="Filename tag for outputs: efficiency_map{tag}.csv/.png. Default '' writes the canonical NSF path; use e.g. '_maf' for the ablation.")
    parser.add_argument("--skip-anchor-validation", action="store_true", help="Skip the anchor-case validation against the NSF cp3 reference CSV (the anchors are NSF-specific; the ablation model is expected to deviate).")
    parser.add_argument("--b-scheme", type=str, default="clinical_sparse",
                        choices=sorted(B_SCHEMES.keys()),
                        help="Acquisition b-value scheme to evaluate on. Must match the scheme the "
                             "model was trained on. 'dense' (16-point) is the acquisition-density "
                             "supplementary experiment; default 'clinical_sparse' (8-point).")
    parser.add_argument("--log-dstar", dest="log_dstar", default=None,
                        action="store_true",
                        help="Force log10(Dstar) reparameterization instead of auto-detecting "
                             "it from the prior support. Needed for priors whose support is not "
                             "a simple box (e.g. the tissue_dstar ablation's MultipleIndependent "
                             "prior, where auto-detection cannot read a lower bound). Default: "
                             "auto-detect (unchanged for setB/MAF/dense).")
    parser.add_argument("--no-log-dstar", dest="log_dstar", action="store_false",
                        help="Force linear Dstar (override auto-detection).")
    args = parser.parse_args()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 80)
    print("Phase E: NPE vs CRLB vs NLLS Efficiency Map")
    print("=" * 80)

    model_path = args.model
    if not os.path.exists(model_path):
        alt_path = "npe/" + os.path.basename(model_path) if not model_path.startswith("npe/") else os.path.basename(model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find model at {model_path}")

    # Resolve pickling namespace dynamically
    sys.modules['__main__'].SNRWrapperEmbedding = SNRWrapperEmbedding

    print(f"Loading model from {model_path}...")
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)

    if args.log_dstar is None:
        log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
        print(f"Auto-detected log_dstar = {log_dstar}")
    else:
        log_dstar = bool(args.log_dstar)
        print(f"Using forced log_dstar = {log_dstar} (CLI override)")

    prior, _, _ = get_processed_prior(device="cpu", log_dstar=log_dstar)
    
    # Simulators
    scheme_bvals = B_SCHEMES[args.b_scheme]
    print(f"Evaluating on b-scheme '{args.b_scheme}' ({scheme_bvals.size} b-values).")
    sim_clean = IVIMNPESimulator(representation="set", clean=True,
                                 active_bvals=scheme_bvals, seed=seed)
    sim_noisy = IVIMNPESimulator(representation="set", clean=False,
                                 active_bvals=scheme_bvals, seed=seed)
    active_bvals = sim_clean.active_bvals

    # Define the 2048-point dense grid
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

    print(f"Dense parameter grid size: {n_grid_points} points")
    print(f"SNR levels: {snr_levels}")
    print(f"Noise realizations per point: N = {n_repeats}")
    print(f"Total points x SNR: {n_grid_points * len(snr_levels)} = 2048")

    # Outputs structure
    results = []
    
    num_cores = os.cpu_count() or 4
    print(f"Using {num_cores} CPU cores for NLLS fitting.")

    start_wall_clock = time.perf_counter()

    # Loop over SNR levels
    for snr in snr_levels:
        t0_snr = time.perf_counter()
        print(f"\nProcessing SNR = {snr:.1f}...")

        # 1. Clean observations and NPE claimed posterior SDs
        obs_clean, snr_ctx_clean = sim_clean(theta_grid, snr=snr) # (512, K, 2)
        x_clean = pack_x(obs_clean, snr_ctx_clean, "set")
        
        with torch.no_grad():
            # (1000, 512, 3)
            samples_clean = posterior.sample_batched((1000,), x=x_clean)
        
        samples_clean_abs = invert_theta(samples_clean, log_dstar=log_dstar)
        samples_clean_disp = to_display(samples_clean_abs).numpy() # (1000, 512, 3)
        
        npe_post_sd_snr = np.std(samples_clean_disp, axis=0) # (512, 3)
        p5 = np.percentile(samples_clean_disp, 5, axis=0)
        p95 = np.percentile(samples_clean_disp, 95, axis=0)
        npe_post_w90_snr = p95 - p5 # (512, 3)

        # 2. Noisy observations and Rician noise realizations
        clean_signals = obs_clean[..., 1].cpu().numpy() # (512, K)
        tiled_clean = np.repeat(clean_signals[:, None, :], n_repeats, axis=1) # (512, N, K)
        flat_clean = tiled_clean.reshape(-1, len(active_bvals)) # (512*N, K)
        flat_noisy = add_rician_noise(flat_clean, snr, S0=1.0, rng=np.random.default_rng(seed)) # (512*N, K)

        # 3. Fit repeats using NLLS in parallel
        print(f"  Fitting {len(flat_noisy)} realizations via NLLS...")
        task_args = [(active_bvals, flat_noisy[i]) for i in range(len(flat_noisy))]
        with multiprocessing.Pool(processes=num_cores) as pool:
            nlls_fits_abs = pool.map(_nlls_worker, task_args)
        nlls_fits_abs = np.array(nlls_fits_abs).reshape(n_grid_points, n_repeats, 3) # (512, N, 3)
        nlls_fits_disp = nlls_fits_abs * np.array([1000.0, 1000.0, 1.0])
        nlls_emp_sd_snr = np.nanstd(nlls_fits_disp, axis=1) # (512, 3)

        # 4. NPE posterior sampling on the same noisy realizations
        print(f"  Sampling posteriors for {len(flat_noisy)} noisy realizations...")
        bvals_tiled = np.tile(active_bvals[None, :], (len(flat_noisy), 1))
        obs_torch = torch.as_tensor(np.stack([bvals_tiled, flat_noisy], axis=-1), dtype=torch.float32)
        snr_ctx_torch = torch.full((len(flat_noisy), 1), np.log10(snr), dtype=torch.float32)
        x_torch = pack_x(obs_torch, snr_ctx_torch, "set")
        
        # Chunk to avoid memory overheads
        chunk_size = 5000
        samples_npe_list = []
        for start_idx in range(0, len(x_torch), chunk_size):
            end_idx = min(start_idx + chunk_size, len(x_torch))
            with torch.no_grad():
                chunk_samples = posterior.sample_batched((args.n_noisy_samples,), x=x_torch[start_idx:end_idx])
            samples_npe_list.append(chunk_samples)
        samples_npe = torch.cat(samples_npe_list, dim=1) # (n_noisy_samples, 512*N, 3)
        
        samples_npe_abs = invert_theta(samples_npe, log_dstar=log_dstar)
        samples_npe_disp = to_display(samples_npe_abs).numpy() # (1000, 512*N, 3)
        
        npe_means = np.mean(samples_npe_disp, axis=0) # (512*N, 3)
        npe_means = npe_means.reshape(n_grid_points, n_repeats, 3) # (512, N, 3)
        npe_emp_sd_snr = np.std(npe_means, axis=1) # (512, 3)

        # 5. Compute CRLB, ratios, and classify
        print(f"  Computing CRLB and calculating efficiency ratios...")
        for j in range(n_grid_points):
            theta_j = theta_grid[j]
            crlb_abs = compute_crlb(theta_j, active_bvals, snr)
            crlb_disp = crlb_abs * np.array([1000.0, 1000.0, 1.0])

            # Convert theta to display units
            theta_disp = theta_j * np.array([1000.0, 1000.0, 1.0])

            for p_idx, p_name in enumerate(["D", "Dstar", "f"]):
                c_sd = crlb_disp[p_idx]
                post_sd = npe_post_sd_snr[j, p_idx]
                post_w90 = npe_post_w90_snr[j, p_idx]
                emp_sd = npe_emp_sd_snr[j, p_idx]
                nlls_sd = nlls_emp_sd_snr[j, p_idx]

                if np.isnan(c_sd) or c_sd <= 0:
                    npe_post_ratio = np.nan
                    npe_emp_ratio = np.nan
                    nlls_ratio = np.nan
                    regime = "unknown"
                else:
                    npe_post_ratio = post_sd / c_sd
                    npe_emp_ratio = emp_sd / c_sd
                    nlls_ratio = nlls_sd / c_sd
                    
                    if npe_post_ratio < 0.9:
                        regime = "overconfident"
                    elif npe_post_ratio <= 1.5:
                        regime = "efficient"
                    else:
                        regime = "inefficient"

                results.append({
                    "D_true": theta_disp[0],
                    "Dstar_true": theta_disp[1],
                    "f_true": theta_disp[2],
                    "snr": snr,
                    "parameter": p_name,
                    "crlb_sd": c_sd,
                    "npe_post_sd": post_sd,
                    "npe_post_w90": post_w90,
                    "npe_emp_sd": emp_sd,
                    "nlls_emp_sd": nlls_sd,
                    "npe_post_ratio": npe_post_ratio,
                    "npe_emp_ratio": npe_emp_ratio,
                    "nlls_ratio": nlls_ratio,
                    "regime": regime
                })

        dt_snr = time.perf_counter() - t0_snr
        print(f"Done SNR = {snr:.1f} in {dt_snr:.2f} seconds.")

    total_wall_clock = time.perf_counter() - start_wall_clock
    print(f"\nGrid sweep completed in {total_wall_clock:.2f} seconds.")

    # Save to CSV
    model_dir = os.path.dirname(model_path) or "."
    csv_path = os.path.join(model_dir, f"efficiency_map{args.out_tag}.csv")
    print(f"\nSaving efficiency map CSV to {csv_path}...")
    with open(csv_path, "w", newline="") as f:
        f.write("# Efficiency Map for Biexponential IVIM model (Phase E)\n")
        f.write("# All standard deviations (SD) and true values are in display units:\n")
        f.write("#   D and Dstar: 1e-3 mm^2/s (unscaled absolute multiplied by 1000)\n")
        f.write("#   f: unitless (unscaled)\n")
        f.write("# Ratios are computed relative to the analytical CRLB SD floor:\n")
        f.write("#   npe_post_ratio = npe_post_sd / crlb_sd  (claimed precision efficiency)\n")
        f.write("#   npe_emp_ratio  = npe_emp_sd  / crlb_sd  (achieved estimator efficiency)\n")
        f.write("#   nlls_ratio     = nlls_emp_sd / crlb_sd  (baseline efficiency)\n")
        f.write("# Regimes classified by npe_post_ratio:\n")
        f.write("#   overconfident: ratio < 0.9\n")
        f.write("#   efficient:     0.9 <= ratio <= 1.5\n")
        f.write("#   inefficient:   ratio > 1.5\n")
        
        writer = csv.DictWriter(f, fieldnames=[
            "D_true", "Dstar_true", "f_true", "snr", "parameter",
            "crlb_sd", "npe_post_sd", "npe_post_w90", "npe_emp_sd", "nlls_emp_sd",
            "npe_post_ratio", "npe_emp_ratio", "nlls_ratio", "regime"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved.")

    # Let's count regime stats
    regime_stats = {
        p: {snr: {"overconfident": 0, "efficient": 0, "inefficient": 0} for snr in snr_levels}
        for p in ["D", "Dstar", "f"]
    }
    for row in results:
        p = row["parameter"]
        snr = row["snr"]
        reg = row["regime"]
        if reg in regime_stats[p][snr]:
            regime_stats[p][snr][reg] += 1

    # Print regime report
    print("\n" + "=" * 80)
    print("NPE CLAIMED REGIME DISTRIBUTION PER PARAMETER AND SNR")
    print("=" * 80)
    for p in ["D", "Dstar", "f"]:
        print(f"\nParameter: {p}")
        for snr in snr_levels:
            counts = regime_stats[p][snr]
            total = sum(counts.values())
            over_pct = counts["overconfident"] / total * 100 if total else 0
            eff_pct = counts["efficient"] / total * 100 if total else 0
            ineff_pct = counts["inefficient"] / total * 100 if total else 0
            print(f"  SNR = {snr:3.0f}: Overconfident: {counts['overconfident']:3d} ({over_pct:5.1f}%) | "
                  f"Efficient: {counts['efficient']:3d} ({eff_pct:5.1f}%) | "
                  f"Inefficient: {counts['inefficient']:3d} ({ineff_pct:5.1f}%)")
    print("=" * 80)

    # Validation against cp3_crlb_compare_v2.csv
    cp3_csv_path = os.path.join(model_dir, "cp3_crlb_compare_v2.csv")
    if args.skip_anchor_validation:
        print("\nSkipping anchor-case validation (--skip-anchor-validation).")
    elif os.path.exists(cp3_csv_path):
        print(f"\nValidating anchor cases against {cp3_csv_path}...")
        anchor_truths = [
            {"name": "Parenchyma typical", "D": 1.1e-3, "Dstar": 30.0e-3, "f": 0.15},
            {"name": "High perfusion", "D": 1.3e-3, "Dstar": 50.0e-3, "f": 0.25},
            {"name": "Low-f hard", "D": 0.9e-3, "Dstar": 1.5e-2, "f": 0.10}
        ]
        
        # We need to run the validation check for anchors using a separate loop.
        # To get matching numbers, we run N=500 repeats for anchors.
        print("Running anchor evaluations (N=500, matches validation settings)...")
        anchor_val_results = []
        for truth in anchor_truths:
            theta_abs = np.array([truth["D"], truth["Dstar"], truth["f"]])
            theta_disp = theta_abs * np.array([1000.0, 1000.0, 1.0])
            for snr in snr_levels:
                crlb_abs = compute_crlb(theta_abs, active_bvals, snr)
                crlb_disp = crlb_abs * np.array([1000.0, 1000.0, 1.0])

                # Simulate Rician realizations (N=500)
                clean_sig = sim_clean.clean_signals(theta_abs[None, :], active_bvals).ravel()
                tiled_clean = np.broadcast_to(clean_sig, (500, len(active_bvals)))
                noisy_signals = add_rician_noise(tiled_clean, snr, S0=1.0, rng=np.random.default_rng(seed))

                # Fit NLLS
                task_args = [(active_bvals, noisy_signals[i]) for i in range(len(noisy_signals))]
                with multiprocessing.Pool(processes=num_cores) as pool:
                    nlls_fits = pool.map(_nlls_worker, task_args)
                nlls_fits = np.array(nlls_fits) * np.array([1000.0, 1000.0, 1.0])
                nlls_sd_disp = np.nanstd(nlls_fits, axis=0)

                # NPE Posterior SD (average of SDs across realizations, which is what CP3 reports)
                obs_torch = torch.as_tensor(np.stack([np.broadcast_to(active_bvals, (500, len(active_bvals))), noisy_signals], axis=-1), dtype=torch.float32)
                snr_ctx_torch = torch.full((500, 1), np.log10(snr), dtype=torch.float32)
                x_torch = pack_x(obs_torch, snr_ctx_torch, "set")
                
                # Draw samples in chunks to avoid memory issues
                samples_npe_list = []
                for start_idx in range(0, len(x_torch), 100):
                    end_idx = min(start_idx + 100, len(x_torch))
                    with torch.no_grad():
                        chunk_samples = posterior.sample_batched((1000,), x=x_torch[start_idx:end_idx])
                    samples_npe_list.append(chunk_samples)
                samples_npe = torch.cat(samples_npe_list, dim=1) # (1000, 500, 3)
                
                samples_npe_abs = invert_theta(samples_npe, log_dstar=log_dstar)
                samples_npe_disp = to_display(samples_npe_abs).numpy()
                npe_sds_per_repeat = np.std(samples_npe_disp, axis=0) # (500, 3)
                npe_sd_disp = np.mean(npe_sds_per_repeat, axis=0) # (3,)

                for p_idx, p_name in enumerate(["D", "Dstar", "f"]):
                    anchor_val_results.append({
                        "anchor_case": truth["name"],
                        "snr": snr,
                        "parameter": p_name,
                        "crlb_sd": crlb_disp[p_idx],
                        "nlls_empirical_sd": nlls_sd_disp[p_idx],
                        "npe_posterior_sd": npe_sd_disp[p_idx]
                    })

        # Load reference CSV
        ref_data = {}
        with open(cp3_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["anchor_case"], float(r["snr"]), r["parameter"])
                ref_data[key] = {
                    "crlb_sd": float(r["crlb_sd"]),
                    "nlls_empirical_sd": float(r["nlls_empirical_sd"]),
                    "npe_posterior_sd": float(r["npe_posterior_sd"])
                }

        # Compare and print deviations
        print(f"\nValidation comparison (computed vs reference in {cp3_csv_path}):")
        print(f"{'Anchor Case':<20} {'SNR':<5} {'Param':<6} | {'Metric':<10} | {'Computed':<10} {'Reference':<10} {'Dev %':<8}")
        print("-" * 85)
        
        max_deviation = 0.0
        failures = []
        for val in anchor_val_results:
            key = (val["anchor_case"], val["snr"], val["parameter"])
            if key not in ref_data:
                continue
            ref = ref_data[key]
            for metric in ["crlb_sd", "nlls_empirical_sd", "npe_posterior_sd"]:
                comp_val = val[metric]
                ref_val = ref[metric]
                if np.isnan(comp_val) and np.isnan(ref_val):
                    continue
                if ref_val == 0 or np.isnan(ref_val):
                    continue
                dev = abs(comp_val - ref_val) / ref_val
                max_deviation = max(max_deviation, dev)
                
                # Check 5% tolerance
                if dev > 0.05:
                    failures.append((key, metric, comp_val, ref_val, dev))
                
                print(f"{val['anchor_case']:<20} {val['snr']:<5.1f} {val['parameter']:<6} | {metric:<10} | {comp_val:<10.4f} {ref_val:<10.4f} {dev*100:<7.2f}%")
        
        print("-" * 85)
        print(f"Max relative deviation: {max_deviation * 100:.2f}%")
        
        # We will report failures but not raise error for empirical deviations since they are subject to noise.
        # CRLB is deterministic, and NPE posterior is mostly deterministic (except for sampling noise).
        crlb_npe_failures = [f for f in failures if f[1] in ["crlb_sd", "npe_posterior_sd"]]
        if len(crlb_npe_failures) == 0:
            print("Verdict: Validation PASSED! Analytical CRLB and NPE posterior SDs match within 5% tolerance.")
        else:
            print(f"Warning: {len(crlb_npe_failures)} CRLB/NPE values deviated by >5%.")
    else:
        print(f"Reference file {cp3_csv_path} not found. Skipping validation.")

    # 6. Generate static faceted plot
    png_path = os.path.join(model_dir, f"efficiency_map{args.out_tag}.png")
    print(f"\nGenerating 3x4 panel grid plot to {png_path}...")
    
    # Select representative D closest to typical parenchyma (1.1e-3)
    D_slice_val = d_grid[np.argmin(np.abs(d_grid - 1.1e-3))] # 1.114e-3
    D_slice_disp = D_slice_val * 1000.0

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
    
    regime_colors = {
        "overconfident": "dodgerblue",
        "efficient": "forestgreen",
        "inefficient": "crimson"
    }
    
    # We will filter results for the nearest D slice
    for p_idx, p_name in enumerate(["D", "Dstar", "f"]):
        for snr_idx, snr in enumerate(snr_levels):
            ax = axes[p_idx, snr_idx]
            
            # Subset results for this SNR, parameter, and nearest D slice
            subset = [
                r for r in results 
                if r["snr"] == snr 
                and r["parameter"] == p_name 
                and abs(r["D_true"] - D_slice_disp) < 1e-5
            ]
            
            # Scatter plot of (f_true, Dstar_true) colored by regime
            f_vals = [r["f_true"] for r in subset]
            dstar_vals = [r["Dstar_true"] for r in subset]
            colors = [regime_colors.get(r["regime"], "gray") for r in subset]
            
            sc = ax.scatter(f_vals, dstar_vals, c=colors, s=80, edgecolors='k', alpha=0.85)
            
            # Calculate mean ratios for annotation
            nlls_ratios = [r["nlls_ratio"] for r in subset if not np.isnan(r["nlls_ratio"])]
            npe_post_ratios = [r["npe_post_ratio"] for r in subset if not np.isnan(r["npe_post_ratio"])]
            
            mean_nlls = np.mean(nlls_ratios) if nlls_ratios else np.nan
            mean_npe = np.mean(npe_post_ratios) if npe_post_ratios else np.nan
            
            # Annotate with mean NLLS and NPE ratios
            ann_text = f"Mean NPE Ratio: {mean_npe:.2f}\nMean NLLS Ratio: {mean_nlls:.2f}"
            ax.text(
                0.05, 0.05, ann_text, 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray")
            )
            
            # Title for columns
            if p_idx == 0:
                ax.set_title(f"SNR = {snr:.0f}", fontsize=14, fontweight='bold')
            
            # Label for rows
            if snr_idx == 0:
                ax.set_ylabel(f"True D* [1e-3 mm^2/s]\n(Param: {p_name})", fontsize=12, fontweight='bold')
                
            if p_idx == 2:
                ax.set_xlabel("True f [-]", fontsize=12, fontweight='bold')
                
            ax.grid(True, alpha=0.2)
            ax.set_xlim([0.0, 0.5])
            ax.set_ylim([0.0, 130.0])

    # Add regime legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=regime_colors["overconfident"], markersize=10, label="Overconfident (ratio < 0.9)", markeredgecolor='k'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=regime_colors["efficient"], markersize=10, label="Efficient (0.9 <= ratio <= 1.5)", markeredgecolor='k'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=regime_colors["inefficient"], markersize=10, label="Inefficient (ratio > 1.5)", markeredgecolor='k')
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=12)

    fig.suptitle(f"NPE Claimed Uncertainty Regime map\n(D slice ≈ {D_slice_disp:.2f} um^2/ms, N={n_repeats})", fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to {png_path}")
    print("\nScript completed successfully!")


if __name__ == "__main__":
    main()
