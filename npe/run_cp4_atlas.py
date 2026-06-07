"""
run_cp4_atlas.py
================
Checkpoint 4: Identifiability Atlas generation.
Computes shrinkage, correlation, and mono-exponential collapse metrics over a
dense parameter-SNR grid, classifies each point, and saves the data products.
Generates a faceted 300-dpi plot representing recoverability regions with CRLB contour overlays.
"""
from __future__ import annotations

import os
import csv
import numpy as np
import torch

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from npe_prior import get_processed_prior, PRIOR_LOW, PRIOR_HIGH, to_display, DISPLAY_SCALE, invert_theta
from npe_simulator import IVIMNPESimulator
from train_npe import pack_x, SNRWrapperEmbedding
from run_cp3_validation import compute_crlb


def compute_metrics(
    samples: torch.Tensor,
    prior_low: torch.Tensor,
    prior_high: torch.Tensor
) -> tuple[np.ndarray, float, float]:
    """
    Computes shrinkage, correlation, and collapse metrics for a set of posterior samples.
    """
    prior_90_width = 0.9 * (prior_high - prior_low)
    
    p5 = torch.quantile(samples, 0.05, dim=0)
    p95 = torch.quantile(samples, 0.95, dim=0)
    post_90_width = p95 - p5
    
    shrinkage = (1.0 - (post_90_width / prior_90_width)).numpy()
    
    # f (index 2) vs Dstar (index 1) correlation
    f_samples = samples[:, 2].numpy()
    Dstar_samples = samples[:, 1].numpy()
    if np.std(f_samples) > 1e-10 and np.std(Dstar_samples) > 1e-10:
        correlation = np.corrcoef(f_samples, Dstar_samples)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
        
    # Mono-exp collapse indicator: mass on f < 0.025
    collapse_indicator = float(torch.mean((samples[:, 2] < 0.025).float()).item())
    
    return shrinkage, correlation, collapse_indicator


def classify_point(shrinkage_Dstar: float, correlation: float, collapse_indicator: float) -> str:
    """
    Classifies a grid point into one of three regimes:
    - mono-exp collapse
    - f-D* trade-off
    - recoverable
    """
    if collapse_indicator > 0.5:
        return "mono-exp collapse"
    elif abs(correlation) > 0.5 or shrinkage_Dstar < 0.5:
        return "f-D* trade-off"
    else:
        return "recoverable"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run CP4 Identifiability Atlas.")
    parser.add_argument("--model", type=str, default="npe/npe_posterior_setB.pt", help="Path to model file.")
    parser.add_argument("--suffix", type=str, default="_v2", help="Suffix for output files.")
    args = parser.parse_args()

    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("=" * 80)
    print("Phase E - Checkpoint 4: Identifiability Atlas")
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

    print(f"Loading setB model from {model_path}...")
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)

    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    print(f"Auto-detected log_dstar = {log_dstar}")

    prior, _, _ = get_processed_prior(device="cpu", log_dstar=log_dstar)
    # Clean simulator for the atlas ground truths
    sim_clean = IVIMNPESimulator(representation="set", clean=True, seed=seed)
    active_bvals = sim_clean.active_bvals

    # -- 1. Define Dense Grid --
    # Absolute ranges
    d_grid = np.linspace(0.6e-3, 2.4e-3, 8)
    dstar_grid = np.linspace(10e-3, 120e-3, 8)
    f_grid = np.linspace(0.02, 0.45, 8)
    snr_levels = [10.0, 20.0, 50.0, 100.0]

    # Prior limits as tensors
    prior_low_t = torch.tensor(PRIOR_LOW, dtype=torch.float32)
    prior_high_t = torch.tensor(PRIOR_HIGH, dtype=torch.float32)

    # Pre-build coordinate grid
    theta_grid = []
    for d in d_grid:
        for dstar in dstar_grid:
            for f in f_grid:
                theta_grid.append([d, dstar, f])
    theta_grid = np.array(theta_grid)
    n_grid_points = len(theta_grid)
    
    print(f"Dense parameter grid size: {n_grid_points} points")
    print(f"SNR levels: {snr_levels}")
    print(f"Total simulations to evaluate: {n_grid_points * len(snr_levels)}")

    atlas_data = []

    # -- 2. Run Grid Sweep --
    for snr in snr_levels:
        print(f"Evaluating SNR = {snr:.1f}...")
        
        # Simulate clean observations in one batch for this SNR
        obs, snr_ctx = sim_clean(theta_grid, snr=snr)
        x = pack_x(obs, snr_ctx, "set")
        
        # Sample posterior in parallel
        # 1000 samples for each grid point
        with torch.no_grad():
            # shape (1000, n_grid_points, 3)
            samples = posterior.sample_batched((1000,), x=x)
            
        for j in range(n_grid_points):
            theta_j = theta_grid[j]
            samples_j = samples[:, j, :]
            
            # Invert log-transform if needed
            samples_j_abs = invert_theta(samples_j, log_dstar=log_dstar)
            
            # Compute metrics
            shrinkage, correlation, collapse = compute_metrics(
                samples_j_abs,
                prior_low_t,
                prior_high_t
            )
            
            classification = classify_point(
                shrinkage_Dstar=shrinkage[1],
                correlation=correlation,
                collapse_indicator=collapse
            )
            
            atlas_data.append({
                "D_true": theta_j[0],
                "Dstar_true": theta_j[1],
                "f_true": theta_j[2],
                "snr": snr,
                "shrinkage_D": shrinkage[0],
                "shrinkage_Dstar": shrinkage[1],
                "shrinkage_f": shrinkage[2],
                "corr_f_Dstar": correlation,
                "mono_collapse_indicator": collapse,
                "classification": classification
            })

    model_dir = os.path.dirname(model_path)
    if not model_dir:
        model_dir = "."

    # -- 3. Save Machine-Readable Products --
    # Save to NPZ
    npz_path = os.path.join(model_dir, f"atlas{args.suffix}.npz")
    npz_dict = {
        "D_true": np.array([r["D_true"] for r in atlas_data]),
        "Dstar_true": np.array([r["Dstar_true"] for r in atlas_data]),
        "f_true": np.array([r["f_true"] for r in atlas_data]),
        "snr": np.array([r["snr"] for r in atlas_data]),
        "shrinkage_D": np.array([r["shrinkage_D"] for r in atlas_data]),
        "shrinkage_Dstar": np.array([r["shrinkage_Dstar"] for r in atlas_data]),
        "shrinkage_f": np.array([r["shrinkage_f"] for r in atlas_data]),
        "corr_f_Dstar": np.array([r["corr_f_Dstar"] for r in atlas_data]),
        "mono_collapse_indicator": np.array([r["mono_collapse_indicator"] for r in atlas_data]),
        "classification": np.array([r["classification"] for r in atlas_data])
    }
    np.savez(npz_path, **npz_dict)
    print(f"Saved machine-readable atlas to {npz_path}")

    # Save to CSV (with header documentation)
    csv_path = os.path.join(model_dir, f"atlas{args.suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        # Header documentation
        f.write("# Identifiability Atlas for Biexponential IVIM model\n")
        f.write("# D_true/Dstar_true/f_true: Absolute parameters (mm^2/s, mm^2/s, unitless)\n")
        f.write("# snr: Signal-to-Noise ratio\n")
        f.write("# shrinkage_D/Dstar/f: relative marginal width shrinkage (1 - post_90_width / prior_90_width)\n")
        f.write("# corr_f_Dstar: Pearson correlation between f and D* samples\n")
        f.write("# mono_collapse_indicator: Posterior mass on f < 0.025\n")
        f.write("# classification: recoverable, f-D* trade-off, or mono-exp collapse\n")
        
        writer = csv.DictWriter(f, fieldnames=[
            "D_true", "Dstar_true", "f_true", "snr",
            "shrinkage_D", "shrinkage_Dstar", "shrinkage_f",
            "corr_f_Dstar", "mono_collapse_indicator", "classification"
        ])
        writer.writeheader()
        writer.writerows(atlas_data)
    print(f"Saved machine-readable CSV to {csv_path}")

    # -- 4. Generate Static faceted Map (matpotlib PNG) --
    # Select representative D closest to typical parenchyma (1.1e-3 mm^2/s)
    D_slice_val = d_grid[np.argmin(np.abs(d_grid - 1.1e-3))]
    print(f"\nGenerating static faceted map for D slice ≈ {D_slice_val * 1e3:.2f} um^2/ms (actual: {D_slice_val:.6f})")

    fig_atlas, axes_atlas = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    # Markers & colors for classification
    class_markers = {
        "recoverable": ("o", "green", "Recoverable"),
        "f-D* trade-off": ("s", "orange", "f-D* trade-off"),
        "mono-exp collapse": ("^", "red", "mono-exp collapse")
    }

    for idx, snr in enumerate(snr_levels):
        ax = axes_atlas[idx]
        
        # Subset data for this SNR and this D slice
        subset = [r for r in atlas_data if r["snr"] == snr and abs(r["D_true"] - D_slice_val) < 1e-10]
        
        # Plot grid points colored by classification
        for cls, (marker, color, label) in class_markers.items():
            cls_pts = [r for r in subset if r["classification"] == cls]
            if cls_pts:
                # Convert parameters to display units: Dstar * 1000, f unscaled
                f_vals = [r["f_true"] for r in cls_pts]
                dstar_disp = [r["Dstar_true"] * 1000.0 for r in cls_pts]
                
                # Show label only in the first panel to avoid legend duplication
                lbl = label if idx == 0 else ""
                ax.scatter(f_vals, dstar_disp, color=color, marker=marker, s=60, label=lbl, alpha=0.85)

        # -- Compute and Overlay CRLB Boundary Contour --
        # Define a fine grid of (f, Dstar) to compute CRLB
        fine_f = np.linspace(0.01, 0.49, 50)
        fine_dstar = np.linspace(3e-3, 140e-3, 50)
        F_mesh, Dstar_mesh = np.meshgrid(fine_f, fine_dstar)
        
        crlb_mesh = np.zeros(F_mesh.shape)
        for r_m in range(F_mesh.shape[0]):
            for c_m in range(F_mesh.shape[1]):
                theta_m = [D_slice_val, Dstar_mesh[r_m, c_m], F_mesh[r_m, c_m]]
                crlb_abs = compute_crlb(theta_m, active_bvals, snr)
                # Dstar CRLB standard deviation in display units (times 1000)
                crlb_mesh[r_m, c_m] = crlb_abs[1] * 1000.0

        # Draw contour line at CRLB SD = 40.0 display units (representing transition to non-identifiable)
        # We will label the contour
        contour_val = 40.0
        CS = ax.contour(F_mesh, Dstar_mesh * 1000.0, crlb_mesh, levels=[contour_val], colors="blue", linestyles="--", linewidths=1.5)
        if idx == 0:
            # Add proxy handle for legend
            proxy = plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5, label=f"CRLB D* SD = {contour_val}")
            ax.add_line(proxy)
            
        ax.set_title(f"SNR = {snr:.0f}", fontsize=14)
        ax.set_xlabel("True f [-]", fontsize=12)
        ax.grid(True, alpha=0.2)
        
        # Axis bounds matching prior range on display scale
        ax.set_xlim([0.0, 0.5])
        ax.set_ylim([0.0, 150.0])

    axes_atlas[0].set_ylabel("True D* [1e-3 mm^2/s]", fontsize=12)
    # Add legends
    axes_atlas[0].legend(loc="upper right", fontsize=10)
    
    fig_atlas.suptitle(f"Identifiability Atlas Map (D slice ≈ {D_slice_val * 1000:.1f} um^2/ms)", fontsize=16, y=0.98)
    fig_atlas.tight_layout()
    img_path = os.path.join(model_dir, f"cp4_identifiability_atlas{args.suffix}.png")
    fig_atlas.savefig(img_path, dpi=300)
    plt.close(fig_atlas)
    print(f"Identifiability atlas figure saved to {img_path}")
    print("Done!")


if __name__ == "__main__":
    main()
