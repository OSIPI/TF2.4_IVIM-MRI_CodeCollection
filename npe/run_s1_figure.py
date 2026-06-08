"""
run_s1_figure.py
================
Supplementary Figure S1 — Architecture ablation of the D* pointwise
overconfidence: Neural Spline Flow (NSF, the main-result model) vs Masked
Autoregressive Flow (MAF).

Reads the two efficiency-map CSVs produced by ``run_e_efficiency.py`` (NSF and
MAF), restricts to the D* parameter, and reproduces the EXACT transform used by
the main-text Figure 3 (``make_manuscript_figures.make_figure3``): group by SNR
and take the per-SNR MEDIAN of the SD-to-CRLB ratios over the 512-point
parameter grid. All ratios are already in linear display units (D* inverted out
of log space, x1000) in the CSVs, so no further transform is applied here.

    Panel A — claimed ratio  (npe_post_ratio = post_sd / crlb_sd) vs SNR
    Panel B — achieved ratio (npe_emp_ratio  = emp_sd  / crlb_sd) vs SNR

Both panels: two series (NSF, MAF), a horizontal CRLB-floor reference at 1.0,
and a log y-axis (matching Figure 3's log-log axes).

Outputs (to --out-dir, default figures/manuscript):
    figS1_maf_ablation.png  (300 dpi)
    figS1_maf_ablation.pdf
    figS1_maf_ablation.csv  (the plotted per-SNR median values)
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
COLOR_NSF = "#0072b2"   # Blue
COLOR_MAF = "#d55e00"   # Vermillion

# Axis configuration copied from make_manuscript_figures.make_figure3 so the
# supplementary panel reads on the same scale as the main-text D* panel.
X_TICKS = [10, 20, 50, 100]
Y_TICKS = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 3.0, 5.0]
Y_TICK_LABELS = ["0.01", "0.05", "0.1", "0.5", "1.0", "1.5", "3.0", "5.0"]


def load_median_ratios(csv_path: str) -> pd.DataFrame:
    """Load an efficiency-map CSV and return per-SNR median D* ratios.

    Identical transform to Figure 3: read with comment='#', restrict to the D*
    parameter, group by SNR and take the median of npe_post_ratio (claimed) and
    npe_emp_ratio (achieved) over the parameter grid.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Supplementary Figure S1 (NSF vs MAF D* ablation).")
    parser.add_argument("--nsf-csv", type=str, default="npe/efficiency_map.csv",
                        help="NSF efficiency map CSV (main-result model).")
    parser.add_argument("--maf-csv", type=str, default="npe/efficiency_map_maf.csv",
                        help="MAF efficiency map CSV (ablation model).")
    parser.add_argument("--out-dir", type=str, default="figures/manuscript",
                        help="Directory for figS1 outputs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    nsf = load_median_ratios(args.nsf_csv)
    maf = load_median_ratios(args.maf_csv)

    # Align on the common SNR grid (both should be [10, 20, 50, 100]).
    snrs = sorted(set(nsf["snr"]).intersection(set(maf["snr"])))
    if not snrs:
        sys.exit("Error: NSF and MAF CSVs share no common SNR values.")
    nsf = nsf[nsf["snr"].isin(snrs)].set_index("snr").loc[snrs]
    maf = maf[maf["snr"].isin(snrs)].set_index("snr").loc[snrs]

    # ---- tidy table of plotted values -------------------------------------- #
    plotted = pd.DataFrame({
        "snr": snrs,
        "nsf_claimed_ratio": nsf["npe_post_ratio"].values,
        "maf_claimed_ratio": maf["npe_post_ratio"].values,
        "nsf_achieved_ratio": nsf["npe_emp_ratio"].values,
        "maf_achieved_ratio": maf["npe_emp_ratio"].values,
    })

    # ---- figure ------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    panels = [
        ("A", "Claimed uncertainty", "npe_post_ratio",
         "Claimed SD / CRLB  (post_sd / crlb_sd)"),
        ("B", "Achieved scatter", "npe_emp_ratio",
         "Achieved SD / CRLB  (emp_sd / crlb_sd)"),
    ]

    for ax, (tag, title, col, ylabel) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.grid(True, which="both", ls="--", color="#e0e0e0", lw=0.5, zorder=1)

        # CRLB floor reference at 1.0
        ax.axhline(1.0, color="#333333", ls="--", lw=1.2, zorder=2)

        ax.plot(snrs, nsf[col].values, ls="-", marker="o", color=COLOR_NSF,
                lw=2, ms=6, label="Spline Flow (NSF)", zorder=4)
        ax.plot(snrs, maf[col].values, ls="--", marker="s", color=COLOR_MAF,
                lw=2, ms=6, label="MAF", zorder=4)

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
        ax.set_title(f"{title}", color="black", fontsize=12, fontweight="bold", pad=12)
        ax.text(-0.12, 1.03, tag, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="right")
        ax.text(95, 1.15, "CRLB Floor", color="#333333", fontsize=8,
                fontweight="bold", ha="right")

    handles = [
        Line2D([0], [0], color=COLOR_NSF, ls="-", marker="o", lw=2, ms=6, label="Spline Flow (NSF)"),
        Line2D([0], [0], color=COLOR_MAF, ls="--", marker="s", lw=2, ms=6, label="MAF"),
        Line2D([0], [0], color="#333333", ls="--", lw=1.2, label="CRLB Floor (ratio = 1.0)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    fig.suptitle("Supplementary Figure S1 — D* SD-to-CRLB ratio: Spline Flow vs MAF",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    png_path = os.path.join(args.out_dir, "figS1_maf_ablation.png")
    pdf_path = os.path.join(args.out_dir, "figS1_maf_ablation.pdf")
    csv_path = os.path.join(args.out_dir, "figS1_maf_ablation.csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    plotted.to_csv(csv_path, index=False)

    # ---- factual summary --------------------------------------------------- #
    print("=" * 80)
    print("Supplementary Figure S1 — D* SD-to-CRLB ratio (per-SNR median over 512-pt grid)")
    print("=" * 80)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(plotted.to_string(index=False))
    print(f"\nSaved: {png_path}\n       {pdf_path}\n       {csv_path}")

    # Relative divergence MAF vs NSF, per panel and SNR.
    def rel(a, b):
        return np.where(b != 0, (a - b) / b, np.nan)

    claimed_rel = rel(plotted["maf_claimed_ratio"].values, plotted["nsf_claimed_ratio"].values)
    achieved_rel = rel(plotted["maf_achieved_ratio"].values, plotted["nsf_achieved_ratio"].values)

    print("\nMAF-vs-NSF relative difference (per SNR):")
    print(f"{'SNR':>6} | {'claimed Δ%':>11} | {'achieved Δ%':>12}")
    print("-" * 36)
    for i, s in enumerate(snrs):
        print(f"{s:>6.0f} | {claimed_rel[i]*100:>10.1f}% | {achieved_rel[i]*100:>11.1f}%")

    DIVERGE = 0.20  # 20% relative difference flag
    flags = []
    for i, s in enumerate(snrs):
        if abs(claimed_rel[i]) > DIVERGE:
            flags.append(f"claimed ratio at SNR={s:.0f} (MAF {plotted['maf_claimed_ratio'].values[i]:.3f} "
                         f"vs NSF {plotted['nsf_claimed_ratio'].values[i]:.3f}, {claimed_rel[i]*100:+.0f}%)")
        if abs(achieved_rel[i]) > DIVERGE:
            flags.append(f"achieved ratio at SNR={s:.0f} (MAF {plotted['maf_achieved_ratio'].values[i]:.3f} "
                         f"vs NSF {plotted['nsf_achieved_ratio'].values[i]:.3f}, {achieved_rel[i]*100:+.0f}%)")
    print(f"\nDivergence points (>|{DIVERGE*100:.0f}%| relative): "
          + ("; ".join(flags) if flags else "none"))


if __name__ == "__main__":
    main()
