"""
run_s5_figure.py
================
Supplementary Figure S5 — Prior-sensitivity ablation of the D* pointwise
overconfidence: the log-uniform-style prior of the primary model (setB) vs a
tissue-informed prior whose log10(D*) marginal is a truncated Normal centered on
a physiological D*. This is the one design axis the architecture (S1) and
acquisition-density (S2) ablations leave uncontrolled.

Reads the two efficiency-map CSVs produced by ``run_e_efficiency.py`` (the
log-uniform setB map and the tissue-prior map), restricts to D*, and reproduces
the EXACT transform used by main-text Figure 3 / Figures S1-S2: group by SNR and
take the per-SNR MEDIAN of the SD-to-CRLB ratios over the 512-point grid. All
ratios are already in linear display units in the CSVs, so no further transform.

    Panel A — claimed ratio  (npe_post_ratio = post_sd / crlb_sd) vs SNR
    Panel B — achieved ratio (npe_emp_ratio  = emp_sd  / crlb_sd) vs SNR

Both panels: two series (log-uniform, tissue-informed), a CRLB-floor reference at
1.0, log-log axes — matching the S1/S2 house style.

Outputs (to --out-dir, default figures/manuscript):
    figS5_prior_ablation.png  (300 dpi)
    figS5_prior_ablation.pdf
    figS5_prior_ablation.csv  (the plotted per-SNR median values)
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
COLOR_LOGUNIF = "#0072b2"   # Blue   (log-uniform prior — the primary model)
COLOR_TISSUE = "#009e73"    # Green  (tissue-informed prior)

X_TICKS = [10, 20, 50, 100]
Y_TICKS = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 3.0, 5.0]
Y_TICK_LABELS = ["0.01", "0.05", "0.1", "0.5", "1.0", "1.5", "3.0", "5.0"]


def load_median_ratios(csv_path: str) -> pd.DataFrame:
    """Per-SNR median D* claimed/achieved ratios — identical transform to Figure 3."""
    if not os.path.exists(csv_path):
        sys.exit(f"Error: efficiency map not found: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    df["parameter"] = df["parameter"].astype(str)
    sub = df[df["parameter"] == PARAM]
    if sub.empty:
        sys.exit(f"Error: no rows with parameter=={PARAM!r} in {csv_path}")
    med = (sub.groupby("snr")[["npe_post_ratio", "npe_emp_ratio"]]
              .median().reset_index().sort_values("snr"))
    return med


def below_floor(csv_path: str):
    """D* below-floor fraction (claimed ratio < 0.9): overall and per SNR."""
    df = pd.read_csv(csv_path, comment="#")
    df["parameter"] = df["parameter"].astype(str)
    sub = df[df["parameter"] == PARAM]
    overall = float((sub["npe_post_ratio"] < 0.9).mean() * 100)
    per = (sub.assign(bf=sub["npe_post_ratio"] < 0.9)
              .groupby("snr")["bf"].mean().mul(100).sort_index())
    return overall, per


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Supplementary Figure S5 (prior-sensitivity D* ablation).")
    ap.add_argument("--loguniform-csv", default="npe/efficiency_map.csv",
                    help="Log-uniform-prior efficiency map (the primary setB model).")
    ap.add_argument("--tissue-csv", default="npe/efficiency_map_priorAlt.csv",
                    help="Tissue-informed-prior efficiency map (the ablation model).")
    ap.add_argument("--out-dir", default="figures/manuscript", help="Directory for figS5 outputs.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lu = load_median_ratios(args.loguniform_csv)
    ti = load_median_ratios(args.tissue_csv)

    snrs = sorted(set(lu["snr"]).intersection(set(ti["snr"])))
    if not snrs:
        sys.exit("Error: log-uniform and tissue CSVs share no common SNR values.")
    lu = lu[lu["snr"].isin(snrs)].set_index("snr").loc[snrs]
    ti = ti[ti["snr"].isin(snrs)].set_index("snr").loc[snrs]

    plotted = pd.DataFrame({
        "snr": snrs,
        "loguniform_claimed_ratio": lu["npe_post_ratio"].values,
        "tissue_claimed_ratio": ti["npe_post_ratio"].values,
        "loguniform_achieved_ratio": lu["npe_emp_ratio"].values,
        "tissue_achieved_ratio": ti["npe_emp_ratio"].values,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    panels = [("A", "Claimed uncertainty", "npe_post_ratio", "Claimed SD / CRLB  (post_sd / crlb_sd)"),
              ("B", "Achieved scatter", "npe_emp_ratio", "Achieved SD / CRLB  (emp_sd / crlb_sd)")]

    for ax, (tag, title, col, ylabel) in zip(axes, panels):
        ax.set_facecolor("white")
        ax.grid(True, which="both", ls="--", color="#e0e0e0", lw=0.5, zorder=1)
        ax.axhline(1.0, color="#333333", ls="--", lw=1.2, zorder=2)
        ax.plot(snrs, lu[col].values, ls="-", marker="o", color=COLOR_LOGUNIF,
                lw=2, ms=6, label="Log-uniform prior (setB)", zorder=4)
        ax.plot(snrs, ti[col].values, ls="--", marker="D", color=COLOR_TISSUE,
                lw=2, ms=6, label="Tissue-informed prior", zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(8, 120); ax.set_ylim(0.008, 6.0)
        ax.set_xticks(X_TICKS); ax.set_xticklabels([str(t) for t in X_TICKS], fontsize=9)
        ax.set_yticks(Y_TICKS); ax.set_yticklabels(Y_TICK_LABELS, fontsize=9)
        ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=10, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=10.5, labelpad=8)
        ax.set_title(title, color="black", fontsize=12, fontweight="bold", pad=12)
        ax.text(-0.12, 1.03, tag, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="right")
        ax.text(95, 1.15, "CRLB Floor", color="#333333", fontsize=8, fontweight="bold", ha="right")

    handles = [
        Line2D([0], [0], color=COLOR_LOGUNIF, ls="-", marker="o", lw=2, ms=6, label="Log-uniform prior (setB)"),
        Line2D([0], [0], color=COLOR_TISSUE, ls="--", marker="D", lw=2, ms=6, label="Tissue-informed prior"),
        Line2D([0], [0], color="#333333", ls="--", lw=1.2, label="CRLB Floor (ratio = 1.0)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=3, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    fig.suptitle("Supplementary Figure S5 — D* SD-to-CRLB ratio: log-uniform vs tissue-informed prior",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    png_path = os.path.join(args.out_dir, "figS5_prior_ablation.png")
    pdf_path = os.path.join(args.out_dir, "figS5_prior_ablation.pdf")
    csv_path = os.path.join(args.out_dir, "figS5_prior_ablation.csv")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    plotted.to_csv(csv_path, index=False)

    print("=" * 80)
    print("Supplementary Figure S5 — D* SD-to-CRLB ratio (per-SNR median over 512-pt grid)")
    print("=" * 80)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(plotted.to_string(index=False))
    print(f"\nSaved: {png_path}\n       {pdf_path}\n       {csv_path}")

    lu_overall, lu_per = below_floor(args.loguniform_csv)
    ti_overall, ti_per = below_floor(args.tissue_csv)
    print("\n" + "=" * 80)
    print("D* BELOW-FLOOR FRACTION (claimed ratio < 0.9) — the verdict statistic")
    print("=" * 80)
    print(f"{'prior':<26}{'overall':>9}" + "".join(f"{'SNR='+str(int(s)):>10}" for s in snrs))
    print(f"{'log-uniform (setB)':<26}{lu_overall:>8.1f}%" + "".join(f"{lu_per.get(s, float('nan')):>9.1f}%" for s in snrs))
    print(f"{'tissue-informed':<26}{ti_overall:>8.1f}%" + "".join(f"{ti_per.get(s, float('nan')):>9.1f}%" for s in snrs))

    print("\n" + "-" * 80)
    print("VERDICT")
    print("-" * 80)
    if ti_overall >= 35.0:
        print(f"D* below-floor stays HIGH under the tissue-informed prior "
              f"({ti_overall:.1f}% overall vs {lu_overall:.1f}% log-uniform; "
              f"both in/near the 41-69% S1/S2 band).")
        print("=> The below-floor D* overconfidence persists across a substantively different")
        print("   prior: consistent with it being INTRINSIC to the amortized estimator under")
        print("   weak identifiability, not an artifact of the log-uniform prior. Thesis holds.")
    elif ti_overall <= 10.0:
        print(f"D* below-floor COLLAPSES under the tissue-informed prior "
              f"({ti_overall:.1f}% overall vs {lu_overall:.1f}% log-uniform).")
        print("=> The overconfidence is PRIOR-REVERSION, not intrinsic. The manuscript's central")
        print("   claim must be re-scoped: the recommendation becomes 'choose the prior carefully.'")
    else:
        print(f"D* below-floor is INTERMEDIATE under the tissue-informed prior "
              f"({ti_overall:.1f}% overall vs {lu_overall:.1f}% log-uniform).")
        print("=> Partial prior-sensitivity: the overconfidence is neither fully intrinsic nor a")
        print("   clean collapse. Report both fractions; the framing needs nuance, not a binary verdict.")
    print("\nNote (honest caveat): a tissue-informed prior legitimately permits posterior SD below")
    print("the prior-free CRLB (Bayesian shrinkage), so a high below-floor fraction is not, by")
    print("itself, proof of miscalibration. Read it together with the claimed-vs-achieved gap")
    print("(Panel A vs B): genuine overconfidence shows claimed << achieved, not merely < CRLB.")


if __name__ == "__main__":
    main()
