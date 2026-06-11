"""
run_s4_figure.py
================
Supplementary Figure S4 — Single-subject in-vivo illustration (abdomen, N = 1).

This is the in-vivo case previously shown as main-text Figure 5, demoted to the
supplement per reviewer request: it is a *qualitative* illustration of the weak-D*
identifiability characterised quantitatively in simulation (main-text Figures
2–3), not evidence, and N = 1 precludes any population inference.

NPE and biexponential-NLLS D* maps are computed for one open IVIM abdominal
acquisition. The reviewer-noted railing of the NLLS reference is reported here as
a *result*: in a large fraction of high-SNR ROI voxels the NLLS D* estimate hits a
fit bound — the per-voxel signature of the weak D* identifiability the paper is
about — and those voxels are excluded before the NPE-vs-NLLS spread comparison so
the reported SD/IQR ratios are not contaminated by the rail.

Panels:
    A — reference b = 0 montage (all ROI-bearing slices), ROI contour overlaid
    B — NLLS D* map  (high-SNR ROI voxels; out-of-range marked)
    C — NPE  D* map  (same voxels, same colour scale)
    D — high-SNR ROI D* distribution (NLLS vs NPE) + railing fraction and the
        all-voxel / non-railed SD and IQR ratios

Inputs (committed): download/Data/abdomen.{nii.gz,bval},
download/Data/mask_abdomen_homogeneous.nii.gz, npe/npe_posterior_setB.pt.

Outputs (to --out-dir, default figures/manuscript):
    figS4_invivo_illustration.png  (300 dpi)
    figS4_invivo_illustration.pdf
    figS4_invivo_illustration.csv  (per-voxel nlls_dstar, npe_dstar, nlls_railed)

Reproduce:
    .venv/bin/python npe/run_s4_figure.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import scipy.optimize as opt
import torch
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import generic_gradient_magnitude, prewitt

sys.path.insert(0, os.path.abspath("npe"))
sys.path.insert(0, os.path.abspath("."))
from npe_prior import invert_theta
from train_npe import pack_x, SNRWrapperEmbedding

# 10-point evaluation b-scheme used by the in-vivo demo.
TARGET_BVALS = np.array([0, 10, 20, 30, 50, 75, 100, 150, 400, 600], dtype=float)
SNR_FLOOR = 8.0
# NLLS D* fit bounds (from fit_biexp_nlls); a voxel within eps of either is "railed".
DSTAR_LOWER_RAIL = 0.0033
DSTAR_UPPER_RAIL = 0.1485


def fit_biexp_nlls(bvals, signal, s0=1.0):
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


def make_montage(volume_3d, slices_to_keep):
    """Tile a list of 2D slices into a single 2D montage (NaN background)."""
    n = len(slices_to_keep)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = volume_3d.shape[0], volume_3d.shape[1]
    montage = np.full((rows * h, cols * w), np.nan)
    for idx, z in enumerate(slices_to_keep):
        r, c = idx // cols, idx % cols
        montage[r * h:(r + 1) * h, c * w:(c + 1) * w] = volume_3d[:, :, z]
    return montage


def iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)


def load_voxels(img_path, mask_path, bval_path):
    bvals_raw = np.loadtxt(bval_path)
    img = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    coords = np.argwhere(mask > 0)
    unique_raw = np.unique(bvals_raw)

    fit_signals, snrs = [], []
    for coord in coords:
        sig = img[coord[0], coord[1], coord[2], :]
        s0 = np.mean(sig[bvals_raw == 0])
        vars_rep = [np.var(sig[bvals_raw == v], ddof=1)
                    for v in unique_raw if v != 0 and len(sig[bvals_raw == v]) > 1]
        if vars_rep and s0 > 0:
            sigma = np.sqrt(np.mean(vars_rep))
            snr = s0 / sigma if sigma > 0 else 30.0
        else:
            snr = 30.0
        snrs.append(snr)
        sub = np.array([np.mean(sig[bvals_raw == b]) for b in TARGET_BVALS])
        fit_signals.append(sub / s0 if s0 > 0 else np.zeros_like(TARGET_BVALS))

    return img, mask, coords, bvals_raw, np.array(fit_signals), np.array(snrs)


def prep_npe(model_path):
    sys.modules["__main__"].SNRWrapperEmbedding = SNRWrapperEmbedding
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)
    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    estimator = posterior.posterior_estimator
    std_mod = estimator.net._embedding_net[0]
    orig_mean = std_mod._mean.clone().numpy()
    orig_std = std_mod._std.clone().numpy()
    b_orig = orig_mean[:, 0]
    new_mean = np.zeros((len(TARGET_BVALS), 3))
    new_std = np.zeros((len(TARGET_BVALS), 3))
    for c in range(3):
        new_mean[:, c] = np.interp(TARGET_BVALS, b_orig, orig_mean[:, c])
        new_std[:, c] = np.interp(TARGET_BVALS, b_orig, orig_std[:, c])
    std_mod._mean = torch.as_tensor(new_mean, dtype=torch.float32)
    std_mod._std = torch.as_tensor(new_std, dtype=torch.float32)
    estimator._condition_shape = torch.Size([len(TARGET_BVALS), 3])
    return posterior, log_dstar


def main():
    ap = argparse.ArgumentParser(description="Build Supplementary Figure S4 (in-vivo abdominal illustration).")
    ap.add_argument("--img", default="download/Data/abdomen.nii.gz")
    ap.add_argument("--mask", default="download/Data/mask_abdomen_homogeneous.nii.gz")
    ap.add_argument("--bval", default="download/Data/abdomen.bval")
    ap.add_argument("--model", default="npe/npe_posterior_setB.pt")
    ap.add_argument("--out-dir", default="figures/manuscript")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img, mask, coords, bvals_raw, fit_signals, snrs = load_voxels(args.img, args.mask, args.bval)
    roi_hi = snrs >= SNR_FLOOR
    snrs_clipped = np.clip(snrs, SNR_FLOOR, 100.0)

    nlls_dstar = np.array([fit_biexp_nlls(TARGET_BVALS, fit_signals[i])[1] for i in range(len(coords))])

    posterior, log_dstar = prep_npe(args.model)
    bt = np.tile(TARGET_BVALS[None, :], (len(coords), 1))
    obs = torch.as_tensor(np.stack([bt, fit_signals], axis=-1), dtype=torch.float32)
    snr_ctx = torch.as_tensor(np.log10(snrs_clipped)[:, None], dtype=torch.float32)
    x = pack_x(obs, snr_ctx, "set")
    with torch.no_grad():
        samples = posterior.sample_batched((100,), x=x, show_progress_bars=False)
    npe_dstar = np.mean(invert_theta(samples, log_dstar=log_dstar).cpu().numpy()[:, :, 1], axis=0)

    nlls_hi, npe_hi = nlls_dstar[roi_hi], npe_dstar[roi_hi]
    railed = (nlls_hi <= DSTAR_LOWER_RAIL) | (nlls_hi >= DSTAR_UPPER_RAIL)
    frac_railed = float(np.mean(railed))
    nlls_nr, npe_nr = nlls_hi[~railed], npe_hi[~railed]

    sd_ratio_all = np.std(npe_hi) / np.std(nlls_hi)
    iqr_ratio_all = iqr(npe_hi) / iqr(nlls_hi)
    sd_ratio_nr = np.std(npe_nr) / np.std(nlls_nr)
    iqr_ratio_nr = iqr(npe_nr) / iqr(nlls_nr)

    # ---- D* maps (high-SNR ROI voxels) ------------------------------------- #
    nlls_map = np.full(mask.shape, np.nan)
    npe_map = np.full(mask.shape, np.nan)
    for i, coord in enumerate(coords):
        if roi_hi[i]:
            nlls_map[tuple(coord)] = nlls_dstar[i]
            npe_map[tuple(coord)] = npe_dstar[i]

    slices_with_roi = np.unique(coords[:, 2])
    b0_idx = int(np.where(bvals_raw == 0)[0][0])
    montage_b0 = make_montage(img[:, :, :, b0_idx], slices_with_roi)
    montage_mask = make_montage(mask, slices_with_roi)
    montage_nlls = make_montage(nlls_map, slices_with_roi)
    montage_npe = make_montage(npe_map, slices_with_roi)

    vmin, vmax = np.percentile(np.concatenate([nlls_hi, npe_hi]), [2, 98])
    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_over("red")
    cmap.set_under("blue")
    cmap.set_bad("black", alpha=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    axes = axes.flatten()

    axes[0].imshow(montage_b0.T, cmap="gray", origin="lower")
    axes[0].contour(montage_mask.T, levels=[0.5], colors="red", linewidths=1)
    axes[0].set_title("A. Reference b = 0 (ROI in red)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im_b = axes[1].imshow(montage_nlls.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("B. NLLS D* map (high-SNR ROI)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im_b, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04,
                 extend="both", label="D* (mm$^2$/s)")

    im_c = axes[2].imshow(montage_npe.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title("C. NPE D* map (same voxels/scale)", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im_c, ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04,
                 extend="both", label="D* (mm$^2$/s)")

    axes[3].hist(nlls_hi, bins=40, alpha=0.55, label="NLLS", color="#0072b2")
    axes[3].hist(npe_hi, bins=40, alpha=0.55, label="NPE", color="#9467bd")
    axes[3].set_title("D. High-SNR ROI D* distribution", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("D* (mm$^2$/s)", fontsize=10)
    axes[3].set_ylabel("Voxel count", fontsize=10)
    textstr = (f"N = 1 subject; {int(np.sum(roi_hi))} high-SNR ROI voxels\n"
               f"NLLS railed: {frac_railed:.1%} (excluded below)\n"
               f"SD ratio NPE/NLLS  — all: {sd_ratio_all:.2f}, non-railed: {sd_ratio_nr:.2f}\n"
               f"IQR ratio NPE/NLLS — all: {iqr_ratio_all:.2f}, non-railed: {iqr_ratio_nr:.2f}")
    axes[3].annotate(textstr, xy=(0.04, 0.64), xycoords="axes fraction",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85), fontsize=9)
    axes[3].legend(loc="upper right", fontsize=9)

    fig.suptitle("Supplementary Figure S4 — Single-subject in-vivo illustration (abdomen, N = 1)",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Qualitative illustration of in-vivo weak-D* identifiability (cf. Figures 2–3); "
             "not used for inference. NLLS railing is itself the per-voxel signature of that "
             "weak identifiability and is excluded before the spread comparison.",
             ha="center", fontsize=8.5, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    png = os.path.join(args.out_dir, "figS4_invivo_illustration.png")
    pdf = os.path.join(args.out_dir, "figS4_invivo_illustration.pdf")
    csv_path = os.path.join(args.out_dir, "figS4_invivo_illustration.csv")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nlls_dstar", "npe_dstar", "nlls_railed"])
        for n, p, r in zip(nlls_hi, npe_hi, railed):
            w.writerow([f"{n:.6f}", f"{p:.6f}", int(r)])

    print("=" * 78)
    print("Supplementary Figure S4 — in-vivo abdominal illustration (N = 1)")
    print("=" * 78)
    print(f"high-SNR ROI voxels (SNR >= {SNR_FLOOR:.0f}): {int(np.sum(roi_hi))} / {len(coords)}")
    print(f"NLLS D* boundary-railed: {frac_railed:.1%}")
    print(f"SD  ratio NPE/NLLS  — all {sd_ratio_all:.2f}, non-railed {sd_ratio_nr:.2f}")
    print(f"IQR ratio NPE/NLLS  — all {iqr_ratio_all:.2f}, non-railed {iqr_ratio_nr:.2f}")
    print(f"\nSaved: {png}\n       {pdf}\n       {csv_path}")


if __name__ == "__main__":
    main()
