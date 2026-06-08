"""
make_manuscript_figures.py
==========================
Phase E close-out script. Generates publication-style figures (Figures 1-4)
and the regime fractions CSV from calib_w3.csv, npe/efficiency_map.csv,
npe/f1_misspecification.csv, and npe/f2_realdata.csv.

Usage:
    python3 make_manuscript_figures.py
"""
from __future__ import annotations
import io
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Define paths
CALIB_CSV = "calib_w3.csv"
EFF_CSV = "npe/efficiency_map.csv"
OUT_DIR = "figures/manuscript"
# Figure 4 (robustness + real-data) inputs — committed data products
F1_CSV = "npe/f1_misspecification.csv"
F2_CSV = "npe/f2_realdata.csv"
REGIME_CSV = os.path.join(OUT_DIR, "regime_fractions.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    """Verify input CSVs exist and load them."""

    if not os.path.exists(CALIB_CSV):
        # Check if it exists in the worktree directory and copy if so
        worktree_path = ".claude/worktrees/heuristic-pike-6fbe7a/calib_w3.csv"
        if os.path.exists(worktree_path):
            import shutil
            shutil.copy(worktree_path, CALIB_CSV)
            print(f"Copied {CALIB_CSV} from worktree directory to workspace root.")
        else:
            sys.exit(f"Error: {CALIB_CSV} not found in root or worktree.")
            
    if not os.path.exists(EFF_CSV):
        sys.exit(f"Error: {EFF_CSV} not found. Ensure NPE efficiency map is present.")
        
    df_calib = pd.read_csv(CALIB_CSV)
    df_calib["param"] = df_calib["param"].astype(str)
    
    # Read efficiency map skipping comment lines starting with #
    df_eff = pd.read_csv(EFF_CSV, comment="#")
    df_eff["parameter"] = df_eff["parameter"].astype(str)
    
    return df_calib, df_eff

# Define Okabe-Ito colorblind-safe palette for parameters
COLOR_D = "#0072b2"      # Blue
COLOR_DSTAR = "#d55e00"  # Vermillion (Red-Orange)
COLOR_F = "#009e73"      # Bluish Green

PARAM_COLORS = {"D": COLOR_D, "Dstar": COLOR_DSTAR, "f": COLOR_F}
PARAM_LABELS = {"D": "D", "Dstar": "D*", "f": "f"}

# Define IBM Design Language colorblind-safe palette for estimators (Figure 3)
COLOR_NPE_CLAIMED = "#785ef0"    # Purple
COLOR_NPE_ACHIEVED = "#dc267f"   # Magenta/Pink
COLOR_NLLS_ACHIEVED = "#fe6100"  # Orange/Red

# Diverging cell order (9 configurations)
CELLS = [
    't0_snr10', 't0_snr20', 't0_snr40',
    't1_snr10', 't1_snr20', 't1_snr40',
    't2_snr10', 't2_snr20', 't2_snr40'
]

# Logical row ordering for Heatmap (groups similar paradigms together)
ROW_ORDER = [
    "OGC · bootstrap",
    "OGC_segmented · bootstrap",
    "OGC_Bayesian · laplace",
    "OGC_Bayesian · mcmc",
    "OGC_Bayesian · mcmc_quantile",
    "IVIM_NEToptim · ensemble",
    "IVIM_NEToptim · input_perturbation",
    "Super_IVIM_DC · ensemble",
    "Super_IVIM_DC · input_perturbation",
]

def get_row_name(method: str, generator: str) -> str:
    """Standardize row names for heatmap."""
    m = method.replace("OGC_AmsterdamUMC_", "OGC_").replace("_biexp", "")
    return f"{m} · {generator}"

def get_reliability_series(df: pd.DataFrame, method: str, generator: str, param: str):
    """Compute empirical coverage mean and SE across cells for a series."""
    sub = df[(df["method"] == method) & (df["generator"] == generator) & (df["param"] == param)]
    nominals = sorted(sub["nominal"].unique())
    xs, ys, yerrs = [], [], []
    for nom in nominals:
        vals = sub[np.isclose(sub["nominal"], nom)]["coverage"].dropna().values
        if len(vals) > 0:
            xs.append(nom)
            ys.append(np.mean(vals))
            # Standard error across cells
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            yerrs.append(se)
    return xs, ys, yerrs

def get_gap_matrix(df: pd.DataFrame, param: str):
    """Construct coverage gap matrix for heatmap at nominal 0.95."""
    sub = df[(df["param"] == param) & np.isclose(df["nominal"], 0.95)].copy()
    sub["row"] = sub.apply(lambda r: get_row_name(r["method"], r["generator"]), axis=1)
    
    M = np.full((len(ROW_ORDER), len(CELLS)), np.nan)
    for i, r in enumerate(ROW_ORDER):
        for j, c in enumerate(CELLS):
            v = sub[(sub["row"] == r) & (sub["cell"] == c)]["coverage"]
            if len(v) > 0:
                M[i, j] = float(v.mean()) - 0.95
    return ROW_ORDER, CELLS, M

def make_figure1(df: pd.DataFrame):
    """Figure 1 — Reliability diagrams (from calib_w3.csv). Two panels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    
    # Shared subplot formatting
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, which="both", ls="--", color="#e5e5e5", lw=0.5, zorder=0)
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1.2, zorder=1)
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Nominal Coverage", fontsize=10)
        ax.tick_params(labelsize=9)
    
    ax1.set_ylabel("Empirical Coverage", fontsize=10)
    
    # --- (A) Bayesian paradigm: D* only ---
    method_bay = "OGC_AmsterdamUMC_Bayesian_biexp"
    gens_A = [
        ("laplace", "--", "s", "Laplace SD (D*)"),
        ("mcmc", "-.", "^", "MCMC SD (D*)"),
        ("mcmc_quantile", "-", "D", "MCMC 2.5/97.5 (D*)")
    ]
    
    for gen, ls, mk, label in gens_A:
        xs, ys, yerrs = get_reliability_series(df, method_bay, gen, "Dstar")
        if not xs:
            continue
        
        # MCMC quantile only has nominal 0.95, plot as single marker
        if gen == "mcmc_quantile":
            ax1.errorbar(xs, ys, yerr=yerrs, fmt=mk, color=COLOR_DSTAR,
                         ms=6, capsize=3, elinewidth=1.5, zorder=5, label=label)
        else:
            ax1.errorbar(xs, ys, yerr=yerrs, fmt=mk, ls=ls, color=COLOR_DSTAR,
                         ms=5, lw=2, capsize=3, elinewidth=1.5, zorder=4, label=label)
            
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.text(-0.13, 1.03, "A", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
    ax1.set_title("Bayesian Paradigm (D* Calibration)", fontsize=11, fontweight="bold", pad=10)
    
    # --- (B) Deep-learning paradigm: Super_IVIM_DC network ---
    method_dl = "Super_IVIM_DC"
    gens_B = [
        ("ensemble", ":", "x", "ensemble (epist.)"),
        ("input_perturbation", "-", "o", "input-perturb")
    ]
    
    for gen, ls, mk, gen_lbl in gens_B:
        for param in ("D", "Dstar", "f"):
            xs, ys, yerrs = get_reliability_series(df, method_dl, gen, param)
            if not xs:
                continue
            
            color = PARAM_COLORS[param]
            is_star = (param == "Dstar")
            lw = 2.4 if is_star else 1.2
            alpha = 1.0 if is_star else 0.45
            ms = 6 if is_star else 4
            zorder = 4 if is_star else 3
            
            ax2.errorbar(xs, ys, yerr=yerrs, fmt=mk, ls=ls, color=color,
                         ms=ms, lw=lw, alpha=alpha, capsize=2, elinewidth=1.0, zorder=zorder)
            
    # Subplot B Legends
    param_handles = [
        Line2D([0], [0], color=COLOR_D, lw=1.5, label="D"),
        Line2D([0], [0], color=COLOR_DSTAR, lw=3, label="D* (skew focus)"),
        Line2D([0], [0], color=COLOR_F, lw=1.5, label="f")
    ]
    gen_handles = [
        Line2D([0], [0], color="gray", ls=":", marker="x", label="ensemble (epistemic)"),
        Line2D([0], [0], color="gray", ls="-", marker="o", label="input-perturbation (aleatoric)")
    ]
    
    leg_p = ax2.legend(handles=param_handles, loc="lower right", frameon=True,
                       facecolor="white", edgecolor="#cccccc", fontsize=8.5, title="Parameter")
    leg_p.get_title().set_fontsize(8.5)
    leg_p.get_title().set_weight("bold")
    ax2.add_artist(leg_p)
    
    leg_g = ax2.legend(handles=gen_handles, loc="upper left", frameon=True,
                       facecolor="white", edgecolor="#cccccc", fontsize=8.5, title="Uncertainty Estimator")
    leg_g.get_title().set_fontsize(8.5)
    leg_g.get_title().set_weight("bold")
    
    ax2.text(-0.13, 1.03, "B", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
    ax2.set_title("Deep-Learning Paradigm (Super_IVIM_DC)", fontsize=11, fontweight="bold", pad=10)
    
    fig.suptitle("Reliability Diagrams — Empirical vs Nominal Coverage", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    
    fig.savefig(os.path.join(OUT_DIR, "fig1_reliability_diagrams.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "fig1_reliability_diagrams.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 1 to figures/manuscript/fig1_reliability_diagrams.[png/pdf]")

def make_figure2(df: pd.DataFrame):
    """Figure 2 — Calibration landscape heatmap (from calib_w3.csv)."""
    params = ["D", "Dstar", "f"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300)
    
    vmin, vmax = -0.5, 0.5
    im = None
    
    for ax, param in zip(axes, params):
        rows, cells, M = get_gap_matrix(df, param)
        
        # Set light gray background for missing cells (e.g. ensemble missing cells)
        ax.set_facecolor("#eeeeee")
        
        # Plot heatmap
        im = ax.imshow(M, cmap="RdBu", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        
        # Grid lines between cells
        ax.set_xticks(np.arange(len(cells)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", lw=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        # Axes Ticks & Labels
        ax.set_xticks(range(len(cells)))
        formatted_cells = [c.replace("_snr", "\nSNR") for c in cells]
        ax.set_xticklabels(formatted_cells, fontsize=8)
        
        ax.set_yticks(range(len(rows)))
        if ax == axes[0]:
            ax.set_yticklabels(rows, fontsize=8.5)
        else:
            ax.set_yticklabels([])
            
        # Annotate each cell with numerical coverage gap
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if np.isfinite(val):
                    # Bold font for severe under-coverage gap > 0.25 (i.e. val < -0.25)
                    weight = "bold" if val < -0.25 else "normal"
                    # White text for high contrast on dark red cells
                    text_color = "white" if val < -0.20 or val > 0.40 else "black"
                    
                    text_str = f"{val:+.2f}"
                    ax.text(j, i, text_str, ha="center", va="center",
                            fontsize=7.5, color=text_color, fontweight=weight)
                else:
                    # Explicit label for missing cells
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=7.5, color="#888888")
        
        # Visual highlighting of D* as the visually dominant panel
        if param == "Dstar":
            ax.set_title("D* (Skew Focus)", color=COLOR_DSTAR, fontsize=12, fontweight="bold", pad=12)
            # Thicker vermillion border
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_DSTAR)
                spine.set_linewidth(2.5)
            # Panel B
            ax.text(-0.05, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
        elif param == "D":
            ax.set_title("D (Diffusion)", color="black", fontsize=11, fontweight="bold", pad=12)
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(1.0)
            # Panel A
            ax.text(-0.25, 1.03, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
        else: # f
            ax.set_title("f (Perfusion Fraction)", color="black", fontsize=11, fontweight="bold", pad=12)
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(1.0)
            # Panel C
            ax.text(-0.05, 1.03, "C", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
            
        ax.set_xlabel("Voxel Configuration (Cell)", fontsize=9.5, labelpad=8)
        
    # Shared colorbar on the right
    fig.subplots_adjust(right=0.85, wspace=0.15)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Coverage Gap (Empirical − Nominal 0.95)", fontsize=10, labelpad=10)
    cbar.ax.tick_params(labelsize=8.5)
    
    fig.suptitle("Calibration Landscape Heatmap — Coverage Gap at Nominal 0.95",
                 fontsize=14, fontweight="bold", y=0.98)
    
    fig.savefig(os.path.join(OUT_DIR, "fig2_calibration_heatmap.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig2_calibration_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 2 to figures/manuscript/fig2_calibration_heatmap.[png/pdf]")

def make_figure3(df_eff: pd.DataFrame):
    """Figure 3 — Estimator efficiency audit vs CRLB Floor."""
    # Compute medians by parameter and snr
    df_med = df_eff.groupby(["parameter", "snr"])[["npe_post_ratio", "npe_emp_ratio", "nlls_ratio"]].median().reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=300)
    
    params = ["D", "Dstar", "f"]
    
    # Tick definitions
    x_ticks = [10, 20, 50, 100]
    x_tick_labels = ["10", "20", "50", "100"]
    y_ticks = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 3.0, 5.0]
    y_tick_labels = ["0.01", "0.05", "0.1", "0.5", "1.0", "1.5", "3.0", "5.0"]
    
    for ax, param in zip(axes, params):
        ax.set_facecolor("white")
        
        # 1. Faint horizontal bands for efficiency regimes
        # Overconfident/biased: ratio < 0.9
        ax.axhspan(0.001, 0.9, color="#ffebee", alpha=0.45, zorder=0)
        # Efficient: 0.9 <= ratio <= 1.5
        ax.axhspan(0.9, 1.5, color="#e8f5e9", alpha=0.45, zorder=0)
        # Inefficient: ratio > 1.5
        ax.axhspan(1.5, 10.0, color="#eef2f7", alpha=0.45, zorder=0)
        
        # Grid lines
        ax.grid(True, which="both", ls="--", color="#e0e0e0", lw=0.5, zorder=1)
        
        # 2. Dashed reference line for CRLB floor (ratio = 1.0)
        ax.axhline(1.0, color="#333333", ls="--", lw=1.2, zorder=2)
        
        # Sort and plot series
        sub = df_med[df_med["parameter"] == param].sort_values("snr")
        snrs = sub["snr"].values
        
        # NPE claimed
        ax.plot(snrs, sub["npe_post_ratio"].values, ls="-", marker="o", color=COLOR_NPE_CLAIMED,
                lw=2, ms=6, label="NPE claimed (post_sd/crlb_sd)", zorder=4)
        # NPE achieved
        ax.plot(snrs, sub["npe_emp_ratio"].values, ls="--", marker="s", color=COLOR_NPE_ACHIEVED,
                lw=2, ms=6, label="NPE achieved (emp_sd/crlb_sd)", zorder=4)
        # NLLS achieved
        ax.plot(snrs, sub["nlls_ratio"].values, ls=":", marker="D", color=COLOR_NLLS_ACHIEVED,
                lw=2, ms=6, label="NLLS achieved (nlls_sd/crlb_sd)", zorder=4)
        
        # Log-log axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(8, 120)
        ax.set_ylim(0.008, 6.0)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=9)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=9)
        
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=10, labelpad=8)
        
        # Titles & labels
        if param == "Dstar":
            ax.set_title("D* (Pseudo-diffusion)", color="black", fontsize=12, fontweight="bold", pad=12)
            ax.text(-0.08, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
        elif param == "D":
            ax.set_title("D (Diffusion)", color="black", fontsize=12, fontweight="bold", pad=12)
            ax.text(-0.15, 1.03, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
            ax.set_ylabel("Standard Deviation to CRLB Ratio (SD / CRLB)", fontsize=10.5, labelpad=8)
        else: # f
            ax.set_title("f (Perfusion Fraction)", color="black", fontsize=12, fontweight="bold", pad=12)
            ax.text(-0.08, 1.03, "C", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")
            
        # Label regimes in the background bands
        ax.text(10.5, 3.0, "inefficient (>1.5)", color="#555555", fontsize=8, style="italic", va="center")
        ax.text(10.5, 1.16, "efficient (0.9–1.5)", color="#2e7d32", fontsize=8, style="italic", va="center")
        ax.text(10.5, 0.08, "overconfident / biased (<0.9)", color="#c62828", fontsize=8, style="italic", va="center")
        
        # Label CRLB floor
        ax.text(95, 1.15, "CRLB Floor", color="#333333", fontsize=8, fontweight="bold", ha="right")

    # Legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    crlb_proxy = Line2D([0], [0], color="#333333", ls="--", lw=1.2, label="CRLB Floor (ratio = 1.0)")
    handles.append(crlb_proxy)
    
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=4, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    
    fig.suptitle("Estimator Efficiency Audit vs Analytical CRLB Floor", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig.savefig(os.path.join(OUT_DIR, "fig3_efficiency_audit.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig3_efficiency_audit.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 3 to figures/manuscript/fig3_efficiency_audit.[png/pdf]")

def export_regime_fractions(df_eff: pd.DataFrame):
    """Compute and save regime fractions by parameter x SNR."""
    counts = df_eff.groupby(["parameter", "snr", "regime"]).size().unstack(fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0).reset_index()
    
    # Reorder columns to a logical progression: overconfident, efficient, inefficient
    fractions = fractions[["parameter", "snr", "overconfident", "efficient", "inefficient"]]
    
    csv_path = os.path.join(OUT_DIR, "regime_fractions.csv")
    fractions.to_csv(csv_path, index=False)
    print(f"Saved regime fractions to {csv_path}")
    return fractions

def load_f1_misspecification(path: str):
    """Parse the two-part F1 CSV.

    Part A — held-out-b coverage rows (header
    ``condition,nominal_level,npe_coverage,nlls_coverage,npe_deviation,nlls_deviation``)
    for the three conditions; Part B — alternative-b-scheme regime counts
    (header ``parameter,overconfident_pct,efficient_pct,inefficient_pct``).
    The two tables are split on the ``# Part B`` comment marker and parsed
    independently (comment lines starting with ``#`` are ignored within each).
    """
    with open(path) as fh:
        lines = fh.readlines()
    split_idx = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("# Part B")), None)
    if split_idx is None:
        sys.exit(f"Error: '# Part B' marker not found in {path}.")
    part_a = pd.read_csv(io.StringIO("".join(lines[:split_idx])), comment="#")
    part_a = part_a[part_a["condition"].isin(["baseline", "triexp", "noise_misspec"])].copy()
    part_b = pd.read_csv(io.StringIO("".join(lines[split_idx + 1:])), comment="#")
    return part_a, part_b

# Figure 4 estimator / acquisition-scheme colors (reuse manuscript palette)
COLOR_NPE = COLOR_NPE_ACHIEVED          # Magenta — NPE estimator
COLOR_NLLS = COLOR_NLLS_ACHIEVED        # Orange  — NLLS baseline
COLOR_SCHEME_BASE = COLOR_NPE_CLAIMED   # Purple  — baseline (clinical) b-scheme
COLOR_SCHEME_ALT = COLOR_NPE_ACHIEVED   # Magenta — alternative (optimized) b-scheme

def make_figure4():
    """Figure 4 — Robustness + real-data confirmation of NPE overconfidence.

    Three panels (matplotlib + pandas only; reads committed CSVs):
      (A) Held-out-b coverage on simulated data across three misspecification
          conditions (npe/f1_misspecification.csv, Part A).
      (B) Held-out-b coverage on real data, N=500 (npe/f2_realdata.csv).
      (C) Acquisition shift: baseline vs alternative-scheme overconfident %
          (regime_fractions.csv mean-over-SNR vs f1_misspecification.csv Part B).
    """
    for label, path in (("F1", F1_CSV), ("F2", F2_CSV), ("regime_fractions", REGIME_CSV)):
        if not os.path.exists(path):
            sys.exit(f"Error: {path} not found ({label} input for Figure 4).")

    part_a, part_b = load_f1_misspecification(F1_CSV)
    df_real = pd.read_csv(F2_CSV, comment="#").sort_values("nominal_level")
    df_regime = pd.read_csv(REGIME_CSV)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    for ax in (axA, axB, axC):
        ax.set_facecolor("white")

    # ===== Panel A — Held-out-b coverage (simulated) =====
    axA.grid(True, which="both", ls="--", color="#e5e5e5", lw=0.5, zorder=0)
    axA.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1.2, zorder=1)
    cond_styles = [("baseline", "-"), ("triexp", "--"), ("noise_misspec", ":")]
    for cond, ls in cond_styles:
        sub = part_a[part_a["condition"] == cond].sort_values("nominal_level")
        axA.plot(sub["nominal_level"], sub["npe_coverage"], ls=ls, marker="o",
                 color=COLOR_NPE, ms=5, lw=2, zorder=4)
        axA.plot(sub["nominal_level"], sub["nlls_coverage"], ls=ls, marker="s",
                 color=COLOR_NLLS, ms=5, lw=2, zorder=3)
    axA.set_xlim(0.45, 1.0)
    axA.set_ylim(0.0, 1.0)
    axA.set_xlabel("Nominal Credibility Level", fontsize=10)
    axA.set_ylabel("Empirical Held-out-b Coverage", fontsize=10)
    axA.tick_params(labelsize=9)
    axA.set_title("Held-out-b coverage (simulated)", fontsize=11, fontweight="bold", pad=10)
    axA.text(-0.13, 1.03, "(A)", transform=axA.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="right")
    handles_A = [
        Line2D([0], [0], color=COLOR_NPE, lw=2, marker="o", label="NPE"),
        Line2D([0], [0], color=COLOR_NLLS, lw=2, marker="s", label="NLLS"),
        Line2D([0], [0], color="#555555", lw=1.5, ls="-", label="baseline"),
        Line2D([0], [0], color="#555555", lw=1.5, ls="--", label="triexp"),
        Line2D([0], [0], color="#555555", lw=1.5, ls=":", label="noise_misspec"),
        Line2D([0], [0], color="#777777", lw=1.2, ls="--", label="ideal (y = x)"),
    ]
    axA.legend(handles=handles_A, loc="upper left", frameon=True, facecolor="white",
               edgecolor="#cccccc", fontsize=8)

    # ===== Panel B — Held-out-b coverage (real data, N=500) =====
    axB.grid(True, which="both", ls="--", color="#e5e5e5", lw=0.5, zorder=0)
    axB.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1.2, zorder=1, label="ideal (y = x)")
    axB.plot(df_real["nominal_level"], df_real["npe_coverage"], ls="-", marker="o",
             color=COLOR_NPE, ms=6, lw=2, zorder=4, label="NPE")
    axB.plot(df_real["nominal_level"], df_real["nlls_coverage"], ls="-", marker="s",
             color=COLOR_NLLS, ms=6, lw=2, zorder=3, label="NLLS")
    axB.set_xlim(0.45, 1.0)
    axB.set_ylim(0.0, 1.0)  # full [0,1] so the NPE collapse (~0.01-0.03) is visible
    axB.set_xlabel("Nominal Credibility Level", fontsize=10)
    axB.set_ylabel("Empirical Held-out-b Coverage", fontsize=10)
    axB.tick_params(labelsize=9)
    axB.set_title("Held-out-b coverage (real data, N=500)", fontsize=11, fontweight="bold", pad=10)
    axB.text(-0.13, 1.03, "(B)", transform=axB.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="right")
    axB.legend(loc="upper left", frameon=True, facecolor="white",
               edgecolor="#cccccc", fontsize=9)

    # ===== Panel C — Acquisition shift =====
    params = ["D", "Dstar", "f"]
    labels = [PARAM_LABELS[p] for p in params]
    baseline_over = [df_regime.loc[df_regime["parameter"] == p, "overconfident"].mean() * 100
                     for p in params]
    pb_over = part_b.set_index("parameter")["overconfident_pct"]
    alt_over = [float(pb_over.loc[p]) for p in params]
    x = np.arange(len(params))
    width = 0.38
    axC.grid(True, axis="y", ls="--", color="#e5e5e5", lw=0.5, zorder=0)
    axC.set_axisbelow(True)
    bars_base = axC.bar(x - width / 2, baseline_over, width, color=COLOR_SCHEME_BASE,
                        edgecolor="white", zorder=3, label="Baseline (clinical) b-scheme")
    bars_alt = axC.bar(x + width / 2, alt_over, width, color=COLOR_SCHEME_ALT,
                       edgecolor="white", zorder=3, label="Alternative (optimized) b-scheme")
    for bars in (bars_base, bars_alt):
        for b in bars:
            h = b.get_height()
            axC.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axC.set_xticks(x)
    axC.set_xticklabels(labels)
    # Headroom above the ~100% bars so the legend sits clear of bars/annotations.
    axC.set_ylim(0, 130)
    axC.set_yticks([0, 20, 40, 60, 80, 100])
    axC.set_xlabel("Parameter", fontsize=10)
    axC.set_ylabel("Overconfident grid points (%)", fontsize=10)
    axC.tick_params(labelsize=9)
    axC.set_title("Acquisition shift", fontsize=11, fontweight="bold", pad=10)
    axC.text(-0.13, 1.03, "(C)", transform=axC.transAxes, fontsize=12,
             fontweight="bold", va="top", ha="right")
    axC.legend(loc="upper center", frameon=True, facecolor="white",
               edgecolor="#cccccc", fontsize=8.5)

    fig.suptitle("Robustness and Real-Data Confirmation of NPE Overconfidence",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(os.path.join(OUT_DIR, "fig4_robustness.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig4_robustness.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 4 to figures/manuscript/fig4_robustness.[png/pdf]")

def print_summary(df_calib: pd.DataFrame, df_eff: pd.DataFrame, fractions: pd.DataFrame):
    """Print verification details to stdout."""
    print("\n" + "="*80)
    print("PHASE E FIGURE GENERATION SUMMARY & VERIFICATION")
    print("="*80)
    print(f"Output files generated in: {OUT_DIR}/")
    print(f"  - fig1_reliability_diagrams.png")
    print(f"  - fig1_reliability_diagrams.pdf")
    print(f"  - fig2_calibration_heatmap.png")
    print(f"  - fig2_calibration_heatmap.pdf")
    print(f"  - fig3_efficiency_audit.png")
    print(f"  - fig3_efficiency_audit.pdf")
    print(f"  - fig4_robustness.png")
    print(f"  - fig4_robustness.pdf")
    print(f"  - regime_fractions.csv")
    
    # 1. Sanity check: Figure 3 plotted medians
    print("\n--- Figure 3 plotted medians (groupby(['parameter','snr']).median()) ---")
    df_med = df_eff.groupby(["parameter", "snr"])[["npe_post_ratio", "npe_emp_ratio", "nlls_ratio"]].median()
    print(df_med)
    
    # 2. Sanity check: Figure 1/2 coverage values (Table 1 reconciliation)
    print("\n--- Figure 1/2 coverage values (nominal = 0.95, OGC_AmsterdamUMC_Bayesian_biexp) ---")
    sub_cal = df_calib[(df_calib["method"] == "OGC_AmsterdamUMC_Bayesian_biexp") & 
                       (df_calib["param"] == "Dstar") & 
                       np.isclose(df_calib["nominal"], 0.95)]
    mcmc_q = sub_cal[sub_cal["generator"] == "mcmc_quantile"]["coverage"].mean()
    laplace = sub_cal[sub_cal["generator"] == "laplace"]["coverage"].mean()
    mcmc_sd = sub_cal[sub_cal["generator"] == "mcmc"]["coverage"].mean()
    
    print(f"  MCMC-quantile D* mean coverage @ 0.95: {mcmc_q:.4f} (Expected ≈ 0.94)")
    print(f"  Laplace SD D* mean coverage @ 0.95:    {laplace:.4f} (Expected ≈ 0.30)")
    print(f"  MCMC-SD D* mean coverage @ 0.95:       {mcmc_sd:.4f} (Expected ≈ 0.67)")
    
    # 3. Regime fractions table print
    print("\n--- Part 2 Supporting Table: Regime Fractions by Parameter & SNR ---")
    print(fractions.to_string(index=False))
    print("="*80 + "\n")

def main():
    df_calib, df_eff = load_data()
    print(f"Loaded {CALIB_CSV} ({len(df_calib)} rows) and {EFF_CSV} ({len(df_eff)} rows).")
    
    make_figure1(df_calib)
    make_figure2(df_calib)
    make_figure3(df_eff)
    fractions = export_regime_fractions(df_eff)
    make_figure4()

    print_summary(df_calib, df_eff, fractions)
    print("Figure generation and verification completed successfully.")

if __name__ == "__main__":
    main()
