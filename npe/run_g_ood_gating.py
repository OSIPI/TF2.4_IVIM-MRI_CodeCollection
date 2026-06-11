"""
run_g_ood_gating.py
===================
Phase G: Operationalizing the out-of-distribution (OOD) gate (Checkpoint G1).

The manuscript proposes a deployment-time OOD gate that withholds the amortized
posterior when an input is off-distribution, but leaves the threshold unspecified.
This script makes that gate concrete and quantifies its trade-off on the open
multi-b in-vivo brain dataset, the same data used for the F2 held-out-b check.

Idea
----
At deployment only the *fit* b-values are available, so the gate must be computable
from them alone. For each voxel we compute two deployable discrepancy scores from
the fit b-subset:

  * ``chi2_red``  - reduced chi-square of the biexponential NLLS fit to the fit
                    b-values (classical goodness-of-fit; a model-misfit / OOD signal).
  * ``npe_selfconsistency`` - RMS normalized residual of the *NPE* posterior-
                    predictive distribution against the observed fit signals
                    (does the amortized posterior even explain its own
                    conditioning data?).

The quantity the gate is meant to predict - available here only because this is a
validation dataset with extra b-values - is the held-out-b posterior-predictive
miscalibration: ``heldout_rms_z``, the RMS normalized residual of the NPE
posterior-predictive against the *held-out* b-values (the F2 failure signal).

We then ask, on real voxels:
  1. ROC / AUC: does the deployable score rank voxels by held-out miscalibration?
  2. Calibration recovery: as the gate threshold tightens (retaining only the
     lowest-score voxels), how does the retained set's held-out-b coverage at
     nominal 0.95 and its normalized-residual SD move toward calibration, versus
     the fraction of voxels retained? This is the threshold/recovery trade-off the
     reviewer asked to see, and it yields a concrete recommended operating point.

Outputs (model dir):
  g_ood_gating{tag}.csv         - summary: AUC, rank correlations, recovery curve,
                                  recommended thresholds.
  g_ood_gating_voxels{tag}.csv  - per-voxel scores/targets (for re-analysis).
  g_ood_gating{tag}.png         - score-vs-failure scatter, ROC, recovery curve.
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
import numpy as np
import torch
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_prior import get_processed_prior, invert_theta
from train_npe import pack_x, SNRWrapperEmbedding
from run_f_realdata import (
    fit_biexp_nlls,
    biexp_signal,
    add_rician_noise,
)

PRIOR_BOUNDS = ([0.2e-3, 3.0e-3, 0.0], [3.0e-3, 0.15, 0.5])
NOMINAL_LEVELS = [0.50, 0.68, 0.80, 0.90, 0.95]


# --------------------------------------------------------------------------- #
# Lightweight ROC / AUC (no sklearn dependency required)
# --------------------------------------------------------------------------- #
def roc_curve_auc(scores: np.ndarray, labels: np.ndarray):
    """ROC for a score where HIGHER means more likely positive (=miscalibrated).

    Returns (fpr, tpr, thresholds, auc). AUC via the rank (Mann-Whitney) identity,
    which equals the trapezoidal area and is robust to ties.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    P = int(y.sum())
    N = int((~y).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf]), float("nan")
    tps = np.cumsum(y)
    fps = np.cumsum(~y)
    # keep one point per distinct threshold
    distinct = np.r_[np.diff(s) != 0, True]
    tpr = np.r_[0.0, tps[distinct] / P]
    fpr = np.r_[0.0, fps[distinct] / N]
    thr = np.r_[np.inf, s[distinct]]
    # AUC via rank-sum (mean rank of positives)
    ranks = np.empty(len(scores), float)
    asc = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[asc]
    r = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks over ties
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[asc] = r
    auc = (ranks[labels].sum() - P * (P + 1) / 2.0) / (P * N)
    return fpr, tpr, thr, float(auc)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase G: operationalize the OOD gate on in-vivo data.")
    parser.add_argument("--model", type=str, default="npe/npe_posterior_setB.pt")
    parser.add_argument("--n-voxels", type=int, default=2000, help="Gray-matter voxels to evaluate.")
    parser.add_argument("--n-samples", type=int, default=200, help="Posterior samples per voxel.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-tag", type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print("Checkpoint G1: Operationalizing the OOD gate on the in-vivo brain dataset")
    print("=" * 80)

    # ---- Model -------------------------------------------------------------- #
    model_path = args.model
    if not os.path.exists(model_path):
        alt = "npe/" + os.path.basename(model_path)
        model_path = alt if os.path.exists(alt) else model_path
    sys.modules["__main__"].SNRWrapperEmbedding = SNRWrapperEmbedding
    print(f"Loading model from {model_path}...")
    posterior = torch.load(model_path, map_location="cpu", weights_only=False)
    log_dstar = bool(posterior.prior.support.base_constraint.lower_bound[1] < 0)
    print(f"Auto-detected log_dstar = {log_dstar}")

    estimator = posterior.posterior_estimator
    std_mod = estimator.net._embedding_net[0]
    orig_mean = std_mod._mean.clone()
    orig_std = std_mod._std.clone()

    def set_npe_condition_size(K: int):
        estimator._condition_shape = torch.Size([K, 3])
        std_mod._mean = orig_mean[:K, :]
        std_mod._std = orig_std[:K, :]

    # ---- Data --------------------------------------------------------------- #
    img_path = "download/Data/brain.nii.gz"
    mask_path = "download/Data/brain_mask_gray_matter.nii.gz"
    bval_path = "download/Data/brain.bval"
    if not all(os.path.exists(p) for p in (img_path, mask_path, bval_path)):
        print("Brain dataset not found in download/Data/. Aborting.")
        return

    print("Loading brain dataset and gray matter mask...")
    img = nib.load(img_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    bvals_raw = np.genfromtxt(bval_path, dtype=float)
    unique_bvals, counts = np.unique(bvals_raw, return_counts=True)

    bvals_fit = np.array([0, 50, 200, 600, 1000], float)
    bvals_heldout = np.array([100, 400, 800], float)
    n_dir = int(np.median(counts[unique_bvals > 0]))  # directions per nonzero b (=6 here)
    print(f"Fit b-values     : {bvals_fit}")
    print(f"Held-out b-values: {bvals_heldout}")
    print(f"Directions per nonzero b (trace-averaged): {n_dir} -> SNR_avg = SNR*sqrt({n_dir})")

    coords = np.argwhere(mask > 0)
    n_eval = min(args.n_voxels, len(coords))
    sel = coords[rng.choice(len(coords), size=n_eval, replace=False)]
    print(f"Evaluating {n_eval} of {len(coords)} gray-matter voxels.")

    fit_signals, heldout_signals, snr_single = [], [], []
    for c in sel:
        v = img[c[0], c[1], c[2], :]
        avg = np.array([np.mean(v[bvals_raw == b]) for b in unique_bvals])
        s0 = avg[0]
        norm = avg / s0 if s0 > 0 else avg
        var_rep = [np.var(v[bvals_raw == b], ddof=1) for b in unique_bvals if b != 0]
        sigma_est = np.sqrt(np.mean(var_rep))
        snr = s0 / sigma_est if (sigma_est > 0 and s0 > 0) else 30.0
        snr = float(np.clip(snr, 8.0, 100.0))
        fit_signals.append([norm[np.where(unique_bvals == b)[0][0]] for b in bvals_fit])
        heldout_signals.append([norm[np.where(unique_bvals == b)[0][0]] for b in bvals_heldout])
        snr_single.append(snr)

    fit_signals = np.array(fit_signals)            # (n, 5)
    heldout_signals = np.array(heldout_signals)    # (n, 3)
    snr_single = np.array(snr_single)              # (n,)

    # Drop voxels with any non-finite signal (background/edge voxels can carry NaN).
    finite = (np.isfinite(fit_signals).all(axis=1)
              & np.isfinite(heldout_signals).all(axis=1)
              & np.isfinite(snr_single))
    if not finite.all():
        print(f"Dropping {int((~finite).sum())} voxels with non-finite signals.")
    fit_signals, heldout_signals, snr_single = (
        fit_signals[finite], heldout_signals[finite], snr_single[finite])
    n_eval = int(finite.sum())
    snr_avg = snr_single * np.sqrt(n_dir)          # trace-averaged SNR
    sigma_avg = 1.0 / snr_avg                       # noise SD on normalized averaged signal

    # ---- NLLS fit on fit subset + chi-square goodness of fit (deployable) ---- #
    print("Fitting biexponential NLLS on the fit subset...")
    nlls_fits = np.array([fit_biexp_nlls(bvals_fit, fit_signals[i]) for i in range(n_eval)])
    fit_pred_nlls = np.array([biexp_signal(bvals_fit, *nlls_fits[i]) for i in range(n_eval)])
    resid_fit = fit_signals - fit_pred_nlls
    dof = max(len(bvals_fit) - 3, 1)
    chi2_red = (resid_fit ** 2).sum(axis=1) / (sigma_avg ** 2) / dof   # (n,) deployable score S1

    # ---- NPE posterior on fit subset --------------------------------------- #
    print("Sampling NPE posteriors...")
    obs = np.stack([np.tile(bvals_fit[None, :], (n_eval, 1)), fit_signals], axis=-1)
    x = pack_x(torch.as_tensor(obs, dtype=torch.float32),
               torch.as_tensor(np.log10(snr_avg)[:, None], dtype=torch.float32), "set")
    set_npe_condition_size(len(bvals_fit))
    with torch.no_grad():
        s_npe = posterior.sample_batched((args.n_samples,), x=x, reject_outside_prior=False)
    s_npe = invert_theta(s_npe, log_dstar=log_dstar).cpu().numpy()
    s_npe = np.clip(s_npe, PRIOR_BOUNDS[0], PRIOR_BOUNDS[1])   # (n_samples, n, 3)

    # ---- Posterior-predictive at fit (self-consistency) and held-out b's ---- #
    print("Generating posterior-predictive samples...")
    D = s_npe[..., 0:1]        # (n_samples, n_eval, 1)
    Dstar = s_npe[..., 1:2]
    f = s_npe[..., 2:3]
    snr_b = snr_avg[None, :, None]   # broadcast per-voxel SNR

    def _predict(bvals):
        b = np.asarray(bvals, float)[None, None, :]
        clean = f * np.exp(-b * Dstar) + (1.0 - f) * np.exp(-b * D)   # (n_samples, n_eval, K)
        return add_rician_noise(clean, snr_b, rng=rng)

    pred_fit = _predict(bvals_fit)     # (n_samples, n_eval, 5)
    pred_ho = _predict(bvals_heldout)  # (n_samples, n_eval, 3)

    mean_fit, std_fit = pred_fit.mean(0), pred_fit.std(0) + 1e-9
    mean_ho, std_ho = pred_ho.mean(0), pred_ho.std(0) + 1e-9
    z_fit = (fit_signals - mean_fit) / std_fit
    z_ho = (heldout_signals - mean_ho) / std_ho
    npe_selfconsistency = np.sqrt((z_fit ** 2).mean(axis=1))   # (n,) deployable score S2
    heldout_rms_z = np.sqrt((z_ho ** 2).mean(axis=1))          # (n,) target (held-out failure)

    # Restrict to voxels with finite scores (a few voxels yield degenerate
    # predictive spreads); keep all per-voxel arrays in lock-step.
    good = (np.isfinite(chi2_red) & np.isfinite(npe_selfconsistency)
            & np.isfinite(heldout_rms_z))
    if not good.all():
        print(f"Dropping {int((~good).sum())} voxels with non-finite derived scores.")
    chi2_red, npe_selfconsistency, heldout_rms_z = (
        chi2_red[good], npe_selfconsistency[good], heldout_rms_z[good])
    snr_single, snr_avg = snr_single[good], snr_avg[good]
    heldout_signals = heldout_signals[good]
    z_ho = z_ho[good]
    pred_ho = pred_ho[:, good, :]
    n_eval = int(good.sum())

    # per-voxel held-out coverage at each nominal level
    cover_flags = {}   # level -> (n,) all-held-out-points-inside indicator
    for L in NOMINAL_LEVELS:
        a = 1.0 - L
        lo = np.percentile(pred_ho, a / 2 * 100, axis=0)
        hi = np.percentile(pred_ho, (1 - a / 2) * 100, axis=0)
        inside = (heldout_signals >= lo) & (heldout_signals <= hi)   # (n,3)
        cover_flags[L] = inside.all(axis=1)

    # =====================================================================
    # 1. ROC / ranking: does the deployable score predict held-out failure?
    # =====================================================================
    # Balanced "severely miscalibrated" label = worse-than-median held-out RMS-z.
    med = float(np.median(heldout_rms_z))
    label_bad = heldout_rms_z > med
    fpr1, tpr1, thr1, auc_chi2 = roc_curve_auc(chi2_red, label_bad)
    fpr2, tpr2, thr2, auc_self = roc_curve_auc(npe_selfconsistency, label_bad)
    rho_chi2 = spearman(chi2_red, heldout_rms_z)
    rho_self = spearman(npe_selfconsistency, heldout_rms_z)

    print("\n" + "-" * 80)
    print("RANKING POWER (deployable score -> held-out miscalibration)")
    print(f"  label: held-out RMS-z above median ({med:.2f});  positives={int(label_bad.sum())}/{n_eval}")
    print(f"  chi2_red             : AUC={auc_chi2:.3f}  Spearman rho={rho_chi2:+.3f}")
    print(f"  npe_selfconsistency  : AUC={auc_self:.3f}  Spearman rho={rho_self:+.3f}")

    # pick the stronger deployable score as the gate statistic
    gate_name, gate_score, gate_auc = (
        ("chi2_red", chi2_red, auc_chi2) if auc_chi2 >= auc_self
        else ("npe_selfconsistency", npe_selfconsistency, auc_self))
    print(f"  -> gate statistic: {gate_name} (AUC {gate_auc:.3f})")

    # =====================================================================
    # 2. Calibration-recovery vs gate threshold
    #    Retain voxels with gate_score <= tau; report retained set's held-out
    #    coverage@0.95 and normalized-residual SD vs retained fraction.
    # =====================================================================
    base_cov95 = float(((heldout_signals >= np.percentile(pred_ho, 2.5, axis=0)) &
                        (heldout_signals <= np.percentile(pred_ho, 97.5, axis=0))).mean())
    base_resid_sd = float(np.std(z_ho))
    fractions = np.linspace(0.05, 1.0, 20)

    def recovery_for(score):
        """Retain the lowest-`score` voxels; report held-out coverage@0.95 and
        residual SD of the retained set vs retained fraction."""
        order = np.argsort(score, kind="mergesort")
        rows = []
        for p in fractions:
            k = max(int(round(p * n_eval)), 1)
            keep = order[:k]
            lo = np.percentile(pred_ho[:, keep, :], 2.5, axis=0)
            hi = np.percentile(pred_ho[:, keep, :], 97.5, axis=0)
            inside = (heldout_signals[keep] >= lo) & (heldout_signals[keep] <= hi)
            rows.append({
                "retained_fraction": float(p),
                "threshold": float(score[order[k - 1]]),
                "coverage95_points": float(inside.mean()),
                "heldout_residual_sd": float(np.std(z_ho[keep])),
            })
        return rows

    recovery = recovery_for(gate_score)        # recommended (best-AUC) gate
    recovery_chi2 = recovery_for(chi2_red)     # independent goodness-of-fit gate (sanity)

    print("\n" + "-" * 80)
    print("CALIBRATION RECOVERY vs GATE THRESHOLD")
    print(f"  ungated (all voxels): coverage@0.95 = {base_cov95:.3f}, held-out residual SD = {base_resid_sd:.2f}")
    print(f"  gate statistic = {gate_name}")
    print(f"  {'retain%':>7} {'tau':>9} {'cov@0.95':>9} {'resid_SD':>9}")
    for r in recovery:
        print(f"  {r['retained_fraction']*100:6.0f}% {r['threshold']:9.2f} "
              f"{r['coverage95_points']:9.3f} {r['heldout_residual_sd']:9.2f}")

    # Concrete, non-arbitrary threshold = Youden point of the recommended gate's ROC.
    fpr_g, tpr_g, thr_g = (fpr1, tpr1, thr1) if gate_name == "chi2_red" else (fpr2, tpr2, thr2)
    youden_idx = int(np.argmax(tpr_g - fpr_g))
    youden_thr = float(thr_g[youden_idx])
    keep_y = gate_score <= youden_thr
    retained_at_youden = float(keep_y.mean())
    lo_y = np.percentile(pred_ho[:, keep_y, :], 2.5, axis=0)
    hi_y = np.percentile(pred_ho[:, keep_y, :], 97.5, axis=0)
    cov_at_youden = float(((heldout_signals[keep_y] >= lo_y) & (heldout_signals[keep_y] <= hi_y)).mean())
    sd_at_youden = float(np.std(z_ho[keep_y]))
    print("\nRECOMMENDED THRESHOLD (ROC Youden point of recommended gate)")
    print(f"  {gate_name} <= {youden_thr:.3f}  (TPR={tpr_g[youden_idx]:.3f}, FPR={fpr_g[youden_idx]:.3f})")
    print(f"  retains {retained_at_youden*100:.0f}% of voxels -> coverage@0.95 {cov_at_youden:.3f} "
          f"(ungated {base_cov95:.3f}); residual SD {sd_at_youden:.2f} (ungated {base_resid_sd:.2f})")

    # =====================================================================
    # 3. Save outputs
    # =====================================================================
    model_dir = os.path.dirname(model_path) or "."
    vox_csv = os.path.join(model_dir, f"g_ood_gating_voxels{args.out_tag}.csv")
    with open(vox_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_single", "snr_avg", "chi2_red", "npe_selfconsistency",
                    "heldout_rms_z", "covered95_allheldout"])
        for i in range(n_eval):
            w.writerow([f"{snr_single[i]:.4f}", f"{snr_avg[i]:.4f}", f"{chi2_red[i]:.5f}",
                        f"{npe_selfconsistency[i]:.5f}", f"{heldout_rms_z[i]:.5f}",
                        int(cover_flags[0.95][i])])
    print(f"\nPer-voxel scores -> {vox_csv}")

    sum_csv = os.path.join(model_dir, f"g_ood_gating{args.out_tag}.csv")
    with open(sum_csv, "w", newline="") as f:
        f.write("# Checkpoint G1: OOD gate operating characteristic on in-vivo brain data\n")
        f.write(f"# n_voxels={n_eval}, gate_statistic={gate_name}\n")
        f.write("# Part A: ranking power of deployable scores vs held-out miscalibration\n")
        f.write("score,auc_vs_median_label,spearman_vs_heldout_rmsz\n")
        f.write(f"chi2_red,{auc_chi2:.4f},{rho_chi2:.4f}\n")
        f.write(f"npe_selfconsistency,{auc_self:.4f},{rho_self:.4f}\n")
        f.write("\n# Part B: calibration recovery vs gate threshold "
                f"(ungated coverage@0.95={base_cov95:.4f}, residual_sd={base_resid_sd:.4f})\n")
        w = csv.DictWriter(f, fieldnames=["gate", "retained_fraction", "threshold",
                                          "coverage95_points", "heldout_residual_sd"])
        w.writeheader()
        for r in recovery:
            w.writerow({"gate": gate_name, **{k: f"{v:.5f}" for k, v in r.items()}})
        for r in recovery_chi2:
            w.writerow({"gate": "chi2_red", **{k: f"{v:.5f}" for k, v in r.items()}})
        f.write("\n# Part C: recommended threshold (ROC Youden point of recommended gate)\n")
        f.write("gate_statistic,threshold,tpr,fpr,retained_fraction,coverage95_points,heldout_residual_sd\n")
        f.write(f"{gate_name},{youden_thr:.4f},{tpr_g[youden_idx]:.4f},{fpr_g[youden_idx]:.4f},"
                f"{retained_at_youden:.4f},{cov_at_youden:.4f},{sd_at_youden:.4f}\n")
    print(f"Summary -> {sum_csv}")

    # ---- Figure ------------------------------------------------------------- #
    png = os.path.join(model_dir, f"g_ood_gating{args.out_tag}.png")
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # (A) score vs held-out failure
    ax0 = ax[0]
    sc = ax0.scatter(gate_score, heldout_rms_z, c=snr_single, s=14, cmap="viridis", alpha=0.6)
    ax0.axhline(med, color="k", ls="--", lw=1, alpha=0.6, label=f"median held-out RMS-z={med:.2f}")
    ax0.set_xscale("log")
    ax0.set_xlabel(f"deployable gate score: {gate_name} (fit b-values only)", fontsize=11)
    ax0.set_ylabel("held-out-b RMS normalized residual (failure)", fontsize=11)
    ax0.set_title(f"Gate score vs held-out failure\nAUC={gate_auc:.3f}, Spearman rho="
                  f"{(rho_chi2 if gate_name=='chi2_red' else rho_self):+.3f}", fontsize=12, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax0); cb.set_label("voxel SNR")
    ax0.legend(loc="upper right", fontsize=9); ax0.grid(True, alpha=0.3)

    # (B) ROC
    ax1 = ax[1]
    ax1.plot(fpr1, tpr1, lw=2, label=f"chi2_red (AUC={auc_chi2:.3f})")
    ax1.plot(fpr2, tpr2, lw=2, ls="--", label=f"npe self-consistency (AUC={auc_self:.3f})")
    ax1.plot([0, 1], [0, 1], "k:", alpha=0.5, label="chance")
    ax1.set_xlabel("false positive rate", fontsize=11)
    ax1.set_ylabel("true positive rate", fontsize=11)
    ax1.set_title("ROC: flagging held-out-miscalibrated voxels", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9); ax1.grid(True, alpha=0.3)

    # (C) calibration recovery: coverage@0.95 vs retained fraction, both gates
    ax2 = ax[2]
    fr = [r["retained_fraction"] for r in recovery]
    cov = [r["coverage95_points"] for r in recovery]
    sd = [r["heldout_residual_sd"] for r in recovery]
    cov_chi2 = [r["coverage95_points"] for r in recovery_chi2]
    ax2.plot(fr, cov, "o-", color="dodgerblue", label=f"coverage@0.95 (gate={gate_name})")
    ax2.plot(fr, cov_chi2, "^-", color="darkorange", alpha=0.8, label="coverage@0.95 (gate=chi2_red)")
    ax2.axhline(0.95, color="dodgerblue", ls=":", alpha=0.6, label="nominal 0.95")
    ax2.axhline(base_cov95, color="gray", ls="--", alpha=0.7, label=f"ungated ({base_cov95:.3f})")
    ax2.set_xlabel("retained fraction (gate keeps lowest-score voxels)", fontsize=11)
    ax2.set_ylabel("held-out coverage @0.95 (retained set)", fontsize=11, color="dodgerblue")
    ax2.tick_params(axis="y", labelcolor="dodgerblue")
    ax2.set_ylim(0, 1)
    ax2b = ax2.twinx()
    ax2b.plot(fr, sd, "s--", color="crimson", alpha=0.8, label=f"residual SD (gate={gate_name})")
    ax2b.axhline(1.0, color="crimson", ls=":", alpha=0.5)
    ax2b.set_ylabel("held-out residual SD (1.0 = calibrated)", fontsize=11, color="crimson")
    ax2b.tick_params(axis="y", labelcolor="crimson")
    ax2.axvline(retained_at_youden, color="k", ls="-", alpha=0.35,
                label=f"Youden gate (retain {retained_at_youden*100:.0f}%)")
    ax2.set_title("Calibration recovery vs gate threshold", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)

    fig.suptitle(f"OOD gate operating characteristic on in-vivo brain data (N={n_eval} voxels)",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {png} (+ .pdf)")
    print("\nCheckpoint G1 completed.")


if __name__ == "__main__":
    main()
