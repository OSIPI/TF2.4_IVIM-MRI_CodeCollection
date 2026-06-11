"""
run_s2_figure.py
================
Supplementary Figure S2 — Acquisition-density control of the D* pointwise
overconfidence: the clinically sparse 8-point scheme (the main-result model) vs a
dense 16-point scheme, each with an NPE trained AND evaluated in-distribution on
its own acquisition.

This isolates whether the below-CRLB D* overconfidence is an artefact of
ultra-sparse sampling or an intrinsic prior-reversion of the amortized estimator
under weak identifiability. If the dense-scheme NPE — given far more b-values and
a correspondingly *tighter* CRLB floor — still reports D* uncertainty below that
floor, the overconfidence is a property of the estimator, not of the acquisition.

Reads the two efficiency-map CSVs produced by ``run_e_efficiency.py`` (the
clinical-sparse ``efficiency_map.csv`` and the dense ``efficiency_map_dense.csv``),
restricts to D*, and reproduces the EXACT transform used by main-text Figure 3
(``make_manuscript_figures.make_figure3``): group by SNR and take the per-SNR
MEDIAN of the SD-to-CRLB ratios over the 512-point parameter grid.

    Panel A — claimed ratio  (npe_post_ratio = post_sd / crlb_sd) vs SNR
    Panel B — achieved ratio (npe_emp_ratio  = emp_sd  / crlb_sd) vs SNR

Both panels: two series (clinical-sparse, dense), a CRLB-floor reference at 1.0,
and a log y-axis (matching Figure 3's log-log axes). Each NPE is audited against
the CRLB of *its own* acquisition, so a point below 1.0 is genuine overconfidence
relative to that scheme's information limit.

Outputs (to --out-dir, default figures/manuscript):
    figS2_dense_acquisition.png  (300 dpi)
    figS2_dense_acquisition.pdf
    figS2_dense_acquisition.csv  (the plotted per-SNR median values)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PARAM = "Dstar"

# Series colours (Okabe-Ito, colourblind-safe) — matches the project palette.
COLOR_SPARSE = "#0072b2"   # Blue   (clinical-sparse, 8 b-values; main-result model)
COLOR_DENSE = "#009e73"    # Green  (dense, 16 b-values)

X_TICKS = [10, 20, 50, 100]
Y_TICKS = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 3.0, 5.0]
Y_TICK_LABELS = ["0.01", "0.05", "0.1", "0.5", "1.0", "1.5", "3.0", "5.0"]


def load_median_ratios(csv_path: str) -> pd.DataFrame:
    """Per-SNR median D* ratios — identical transform to main-text Figure 3."""
    if not os.path.exists(csv_path):
        sys.exit(f"Error: efficiency map not found: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    df["parameter"] = df["parameter"].astype(str)
    sub = df[df["parameter"] == PARAM]
    if sub.empty:
        sys.exit(f"Error: no rows with parameter=={PARAM!r} in {csv_path}")
    med = (sub.groupby("snr")[["npe_post_ratio", "npe_emp_ratio"]]
              .median()
              .reset_index()
              .sort_values("snr"))
    return med


def overconfident_fraction(csv_path: str) -> float:
    """Fraction of D* grid points classified 'overconfident' (ratio<0.9)."""
    df = pd.read_csv(csv_path, comment="#")
    sub = df[df["parameter"].astype(str) == PARAM]
    return float((sub["regime"] == "overconfident").mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Supplementary Figure S2 (clinical-sparse vs dense D* control).")
    parser.add_argument("--sparse-csv", type=str, default="npe/efficiency_map.csv",
                        help="Clinical-sparse efficiency map CSV (main-result model).")
    parser.add_argument("--dense-csv", type=str, default="npe/efficiency_map_dense.csv",
                        help="Dense (16-b) efficiency map CSV.")
    parser.add_argument("--out-dir", type=str, default="figures/manuscript")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sparse = load_median_ratios(args.sparse_csv)
    dense = load_median_ratios(args.dense_csv)
    snrs = sorted(set(sparse["snr"]).intersection(set(dense["snr"])))
    if not snrs:
        sys.exit("Error: sparse and dense CSVs share no common SNR values.")
    sparse = sparse[sparse["snr"].isin(snrs)].set_index("snr").loc[snrs]
    dense = dense[dense["snr"].isin(snrs)].set_index("snr").loc[snrs]

    plotted = pd.DataFrame({
        "snr": snrs,
        "sparse_claimed_ratio": sparse["npe_post_ratio"].values,
        "dense_claimed_ratio": dense["npe_post_ratio"].values,
        "sparse_achieved_ratio": sparse["npe_emp_ratio"].values,
        "dense_achieved_ratio": dense["npe_emp_ratio"].values,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    panels = [
        ("A", "Claimed uncertainty", "npe_post_ratio", "Claimed SD / CRLB  (post_sd / crlb_sd)"),
        ("B", "Achieved scatter", "npe_emp_ratio", "Achieved SD / CRLB  (emp_sd / crlb_sd)"),
    ]
    for ax, (tag, title, col, ylabel) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.grid(True, which="both", ls="--", color="#e0e0e0", lw=0.5, zorder=1)
        ax.axhline(1.0, color="#333333", ls="--", lw=1.2, zorder=2)
        ax.plot(snrs, sparse[col].values, ls="-", marker="o", color=COLOR_SPARSE,
                lw=2, ms=6, label="Clinical-sparse (8 b)", zorder=4)
        ax.plot(snrs, dense[col].values, ls="--", marker="D", color=COLOR_DENSE,
                lw=2, ms=6, label="Dense (16 b)", zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(8, 120)
        ax.set_ylim(0.008, 6.0)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([str(t) for t in X_TICKS], fontsize=9)
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels(Y_TICK_LABELS, fontsize=9)
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=10, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=10.5, labelpad=8)
        ax.set_title(title, color="black", fontsize=12, fontweight="bold", pad=12)
        ax.text(-0.12, 1.03, tag, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="right")
        ax.text(95, 1.15, "CRLB Floor", color="#333333", fontsize=8, fontweight="bold", ha="right")

    handles = [
        Line2D([0], [0], color=COLOR_SPARSE, ls="-", marker="o", lw=2, ms=6, label="Clinical-sparse (8 b)"),
        Line2D([0], [0], color=COLOR_DENSE, ls="--", marker="D", lw=2, ms=6, label="Dense (16 b)"),
        Line2D([0], [0], color="#333333", ls="--", lw=1.2, label="CRLB Floor (ratio = 1.0)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    fig.suptitle("Supplementary Figure S2 — D* SD-to-CRLB ratio: clinical-sparse vs dense acquisition",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    png_path = os.path.join(args.out_dir, "figS2_dense_acquisition.png")
    pdf_path = os.path.join(args.out_dir, "figS2_dense_acquisition.pdf")
    csv_path = os.path.join(args.out_dir, "figS2_dense_acquisition.csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    plotted.to_csv(csv_path, index=False)

    # ---- factual summary --------------------------------------------------- #
    of_sparse = overconfident_fraction(args.sparse_csv)
    of_dense = overconfident_fraction(args.dense_csv)
    print("=" * 80)
    print("Supplementary Figure S2 — D* SD-to-CRLB ratio (per-SNR median over 512-pt grid)")
    print("=" * 80)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(plotted.to_string(index=False))
    print(f"\nD* points classified overconfident (ratio<0.9), all SNR:")
    print(f"  clinical-sparse (8 b): {of_sparse*100:.1f}%")
    print(f"  dense          (16 b): {of_dense*100:.1f}%")
    print(f"\nSaved: {png_path}\n       {pdf_path}\n       {csv_path}")
    print("\nInterpretation: a dense-scheme NPE evaluated against the (tighter) dense CRLB floor "
          "remaining below 1.0 demonstrates the D* overconfidence is intrinsic prior-reversion, "
          "not an ultra-sparse-acquisition artefact.")


if __name__ == "__main__":
    main()
