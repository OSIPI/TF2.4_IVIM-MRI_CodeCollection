"""
run_crlb_sampling_bound.py
==========================
Checkpoint 2 (reviewer item 4): a sampling-based CRLB bound on the low-SNR
subset, to test the manuscript's claim that the analytical *Gaussian* CRLB —
least tight where the D* posterior is most skewed (low SNR, weak
identifiability) — under-states the true achievable-variance floor, so the
below-floor D* overconfidence is, if anything, understated rather than created
by the Gaussian approximation.

What it does
------------
On the low-SNR rows (default SNR in {10, 20}) of the *committed* efficiency map
it keeps the published analytical floor and NPE claimed SD untouched (read
straight from `efficiency_map.csv`, so the analytical column is bit-identical to
the figure-3 / S1 numbers), and replaces ONLY the denominator with a Monte-Carlo
floor, in two flavours:

  * crlb_sd_sampled  — EFFICIENT-ESTIMATOR EMPIRICAL VARIANCE (the headline).
                       At each grid theta and SNR, draw R Rician noise
                       realizations and fit each with a *generously bounded*
                       biexponential least-squares estimator; the per-parameter
                       empirical SD across the R fits is the achievable-variance
                       floor. Generous bounds (well outside the prior box) keep
                       the estimator from railing, which would truncate the
                       spread and deflate the floor; the rail rate is reported.

  * crlb_sd_rician   — RICIAN-EXACT FISHER CRLB (estimator-free cross-check).
                       The analytical floor uses a Gaussian approximation of the
                       Rician magnitude likelihood (sigma = S0/SNR). Here we
                       Monte-Carlo the *exact* Rician per-b information
                       g_b = E[(d/dnu log p_Rician)^2] and form
                       CRLB = sqrt(diag(inv(J^T diag(g) J))). This relaxes the
                       Gaussian-noise approximation without any optimisation, so
                       it carries none of the estimator's bias/railing artifacts
                       (but, being a local Fisher bound, it does not capture
                       parameter-space skew — the estimator-variance floor does).

For each floor it forms the directly-comparable ratio  npe_post_sd / floor  and
the below-floor fraction (ratio < 0.9), for D*, overall and per SNR.

Direction-of-error check (the manuscript's argument): a floor that captures the
non-Gaussian geometry should be >= the Gaussian CRLB at low SNR, so the ratio
should be <= the analytical ratio and the below-floor fraction UNCHANGED OR
LARGER. If a sampled fraction comes out SMALLER it is flagged prominently. The
printed caveat: the CRLB bounds only *unbiased* estimators, so a biased low-SNR
estimator can dip below it (a bias artifact, not a refutation) — which is why the
estimator-free Rician column is reported alongside.

Runs under .venv-npe (numpy + scipy, no pandas), matching run_e_efficiency.

Outputs
-------
    crlb_sampling_bound_lowSNR.csv   (per grid-point x SNR, all three floors)
    a printed summary table
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import multiprocessing

import numpy as np
import scipy.optimize as opt
from scipy.special import ive

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ivim_simulator import ivim_signal, add_rician_noise, B_SCHEMES
from run_cp3_validation import compute_crlb, compute_jacobian

PARAM_INDEX = {"D": 0, "Dstar": 1, "f": 2}
DISPLAY = np.array([1000.0, 1000.0, 1.0])

# Generously WIDE fit bounds (well outside the prior box [0.2e-3,3e-3] x
# [3e-3,0.15] x [0,0.5]) so the floor estimator is not railed by the box.
WIDE_LOW = np.array([0.05e-3, 1.0e-3, 0.0])
WIDE_HIGH = np.array([5.0e-3, 0.5, 0.8])
_RAIL_TOL = 1e-3  # relative distance to a bound counted as "railed"


def fit_biexp_wide(bvals: np.ndarray, signal: np.ndarray, S0: float = 1.0) -> np.ndarray:
    """Biexp least-squares with WIDE bounds; returns [D, Dstar, f] absolute, NaN on failure."""
    x0 = [1.6e-3, 50e-3, 0.2]

    def residuals(p):
        D, Dstar, f = p
        pred = S0 * (f * np.exp(-bvals * Dstar) + (1.0 - f) * np.exp(-bvals * D))
        return pred - signal

    try:
        res = opt.least_squares(residuals, x0, bounds=(WIDE_LOW, WIDE_HIGH), method="trf")
        return res.x
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def _fit_worker(args):
    bvals, signal = args
    return fit_biexp_wide(bvals, signal)


def _rail_fraction(fits: np.ndarray, p_idx: int) -> float:
    """Fraction of finite fits whose parameter sits within _RAIL_TOL of a wide bound."""
    col = fits[:, p_idx]
    col = col[np.isfinite(col)]
    if col.size == 0:
        return np.nan
    span = WIDE_HIGH[p_idx] - WIDE_LOW[p_idx]
    near_low = np.abs(col - WIDE_LOW[p_idx]) < _RAIL_TOL * span
    near_high = np.abs(col - WIDE_HIGH[p_idx]) < _RAIL_TOL * span
    return float(np.mean(near_low | near_high))


def sampled_floor(theta_abs, bvals, snr, n_real, seed, pool) -> dict:
    """Empirical SD of the wide-bounds estimator over n_real Rician realizations."""
    clean = ivim_signal(bvals, *theta_abs, S0=1.0)
    rng = np.random.default_rng(seed)
    noisy = add_rician_noise(np.broadcast_to(clean, (n_real, bvals.size)), snr, S0=1.0, rng=rng)
    fits = np.array(pool.map(_fit_worker, [(bvals, noisy[i]) for i in range(n_real)]))
    sd_abs = np.nanstd(fits, axis=0)
    diverged = float(np.mean(~np.isfinite(fits[:, 1])))
    return {"sd_disp": sd_abs * DISPLAY, "rail_dstar": _rail_fraction(fits, 1), "diverged": diverged}


def rician_crlb(theta_abs, bvals, snr, n_real, seed) -> np.ndarray:
    """Exact-Rician Fisher CRLB SD for [D, Dstar, f] (absolute), via MC per-b information.

    g_b = E[(m * I1(z)/I0(z) - nu_b)^2] / sigma^4,  z = m*nu_b/sigma^2,  sigma=S0/SNR.
    CRLB = sqrt(diag(inv(J^T diag(g) J))). Uses exponentially-scaled Bessel ratio
    ive(1,z)/ive(0,z) = I1(z)/I0(z) for numerical stability.
    """
    sigma = 1.0 / snr
    nu = ivim_signal(bvals, *theta_abs, S0=1.0)                 # (K,)
    rng = np.random.default_rng(seed + 777)
    n1 = rng.normal(0.0, sigma, size=(n_real, nu.size))
    n2 = rng.normal(0.0, sigma, size=(n_real, nu.size))
    m = np.sqrt((nu[None, :] + n1) ** 2 + n2 ** 2)              # (R, K)
    z = m * nu[None, :] / sigma ** 2
    ratio = ive(1, z) / ive(0, z)                              # I1/I0, stable
    bracket = (m * ratio - nu[None, :]) ** 2                   # (R, K)
    g = bracket.mean(axis=0) / sigma ** 4                      # (K,) per-b info
    J = compute_jacobian(theta_abs, bvals, S0=1.0)             # (K, 3)
    FIM = J.T @ (g[:, None] * J)
    try:
        var = np.diag(np.linalg.inv(FIM))
        if np.any(var < 0):
            return np.array([np.nan, np.nan, np.nan])
        return np.sqrt(var)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan, np.nan])


def _read_map_rows(path, param, snrs):
    """Read the committed efficiency map (csv with leading # comments) -> list of dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(r for r in f if not r.startswith("#"))
        for r in reader:
            if str(r["parameter"]) != param:
                continue
            if float(r["snr"]) not in snrs:
                continue
            rows.append({
                "D_true": float(r["D_true"]), "Dstar_true": float(r["Dstar_true"]),
                "f_true": float(r["f_true"]), "snr": float(r["snr"]),
                "npe_post_sd": float(r["npe_post_sd"]), "crlb_sd": float(r["crlb_sd"]),
                "npe_post_ratio": float(r["npe_post_ratio"]),
            })
    rows.sort(key=lambda d: (d["snr"], d["Dstar_true"], d["f_true"], d["D_true"]))
    return rows


def _bf(vals):
    """Below-floor fraction (%): ratio < 0.9 over finite entries."""
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return float(np.mean(v < 0.9) * 100) if v.size else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description="Sampling-based CRLB floor on the low-SNR subset.")
    ap.add_argument("--map", default="npe/efficiency_map.csv",
                    help="Committed efficiency map (setB NSF) supplying the analytical floor and NPE claimed SD.")
    ap.add_argument("--snrs", type=float, nargs="+", default=[10.0, 20.0], help="Low-SNR subset.")
    ap.add_argument("--n-real", type=int, default=2000, help="Noise realizations per grid point (R).")
    ap.add_argument("--n-real-rician", type=int, default=8000, help="Realizations for the Rician-FIM MC.")
    ap.add_argument("--b-scheme", default="clinical_sparse", choices=sorted(B_SCHEMES.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process the first N grid points (testing).")
    ap.add_argument("--out", default="npe/crlb_sampling_bound_lowSNR.csv")
    ap.add_argument("--param", default="Dstar", choices=list(PARAM_INDEX.keys()))
    args = ap.parse_args()

    bvals = B_SCHEMES[args.b_scheme]
    p_idx = PARAM_INDEX[args.param]
    snrs = set(args.snrs)

    src = args.map if os.path.exists(args.map) else os.path.join("npe", os.path.basename(args.map))
    rows_in = _read_map_rows(src, args.param, snrs)
    if args.limit > 0:
        rows_in = rows_in[:args.limit]
    if not rows_in:
        sys.exit(f"No {args.param} rows at SNR {sorted(snrs)} in {src}")
    print(f"Loaded {len(rows_in)} {args.param} rows at SNR {sorted(snrs)} from {src}")
    print(f"Sampling floor: R={args.n_real} Rician realizations/point, b-scheme={args.b_scheme}, "
          f"wide D* bounds [{WIDE_LOW[1]*1000:.1f},{WIDE_HIGH[1]*1000:.0f}]e-3")

    num_cores = os.cpu_count() or 4
    out_rows = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        for i, r in enumerate(rows_in):
            theta_abs = np.array([r["D_true"] / 1000.0, r["Dstar_true"] / 1000.0, r["f_true"]])
            snr = r["snr"]
            crlb_chk = compute_crlb(theta_abs, bvals, snr)[p_idx] * DISPLAY[p_idx]
            samp = sampled_floor(theta_abs, bvals, snr, args.n_real, args.seed + i, pool)
            ric = rician_crlb(theta_abs, bvals, snr, args.n_real_rician, args.seed + i)
            f_samp = samp["sd_disp"][p_idx]
            f_ric = ric[p_idx] * DISPLAY[p_idx]
            out_rows.append({
                "D_true": r["D_true"], "Dstar_true": r["Dstar_true"], "f_true": r["f_true"], "snr": snr,
                "npe_post_sd": r["npe_post_sd"],
                "crlb_sd_analytic": r["crlb_sd"],
                "crlb_sd_analytic_recomputed": crlb_chk,
                "crlb_sd_sampled": f_samp,
                "crlb_sd_rician": f_ric,
                "ratio_analytic": r["npe_post_ratio"],
                "ratio_sampled": r["npe_post_sd"] / f_samp if f_samp > 0 else np.nan,
                "ratio_rician": r["npe_post_sd"] / f_ric if np.isfinite(f_ric) and f_ric > 0 else np.nan,
                "rail_rate_dstar": samp["rail_dstar"],
                "diverged_rate": samp["diverged"],
            })
            if (i + 1) % 128 == 0:
                print(f"  ...{i + 1}/{len(rows_in)} points")

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {out_path}")

    # --- analytical self-check (recompute must equal the map) ------------------
    rel = [abs(o["crlb_sd_analytic_recomputed"] - o["crlb_sd_analytic"]) / abs(o["crlb_sd_analytic"])
           for o in out_rows if o["crlb_sd_analytic"]]
    print(f"Analytical-floor self-check: max |recomputed-map|/map = {max(rel)*100:.3f}% "
          f"({'OK' if max(rel) < 0.01 else 'MISMATCH — investigate'})")

    snr_sorted = sorted(snrs)
    col = {k: [o[k] for o in out_rows] for k in out_rows[0]}

    def bf_by_snr(ratio_key):
        per = []
        for s in snr_sorted:
            vals = [o[ratio_key] for o in out_rows if o["snr"] == s]
            per.append(_bf(vals))
        return per

    print("\n" + "=" * 80)
    print(f"{args.param} BELOW-FLOOR FRACTION (claimed SD / floor < 0.9)")
    print("=" * 80)
    header = f"{'floor':<30}{'overall':>9}" + "".join(f"{'SNR='+str(int(s)):>10}" for s in snr_sorted)
    print(header)
    for label, key in [("analytical Gaussian CRLB", "ratio_analytic"),
                       ("sampled (estimator variance)", "ratio_sampled"),
                       ("Rician-exact Fisher CRLB", "ratio_rician")]:
        line = f"{label:<30}{_bf(col[key]):>8.1f}%"
        for v in bf_by_snr(key):
            line += f"{v:>9.1f}%"
        print(line)

    def med(vals):
        v = np.array(vals, float); v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else np.nan

    print("\nMedian floor (display units) and median claimed/floor ratio, per SNR:")
    for s in snr_sorted:
        sel = [o for o in out_rows if o["snr"] == s]
        print(f"  SNR={int(s):>3}: floors analytic={med([o['crlb_sd_analytic'] for o in sel]):7.3f}  "
              f"sampled={med([o['crlb_sd_sampled'] for o in sel]):7.3f}  "
              f"rician={med([o['crlb_sd_rician'] for o in sel]):7.3f}  | "
              f"ratio analytic={med([o['ratio_analytic'] for o in sel]):.3f} "
              f"sampled={med([o['ratio_sampled'] for o in sel]):.3f} "
              f"rician={med([o['ratio_rician'] for o in sel]):.3f}  | "
              f"D* rail={np.nanmean([o['rail_rate_dstar'] for o in sel])*100:4.1f}% "
              f"diverged={np.nanmean([o['diverged_rate'] for o in sel])*100:.1f}%")

    print("\n" + "-" * 80)
    bf_an = _bf(col["ratio_analytic"])
    print("Direction-of-error check (manuscript: sampled floor >= Gaussian -> fraction unchanged/larger):")
    for label, key in [("sampled (estimator var)", "ratio_sampled"), ("Rician-exact Fisher", "ratio_rician")]:
        val = _bf(col[key]); delta = val - bf_an
        if delta >= -0.05:
            print(f"  {label:<26}: {val:.1f}% vs analytic {bf_an:.1f}%  ({delta:+.1f} pts) -> consistent with manuscript")
        else:
            print(f"  ** {label:<23}: {val:.1f}% vs analytic {bf_an:.1f}%  ({delta:+.1f} pts) -> SMALLER; "
                  f"CONTRADICTS stated direction — see caveat **")
    print("Caveat: the CRLB bounds only UNBIASED estimators; at low SNR the estimator is biased,")
    print("so the estimator-variance floor can legitimately dip below the CRLB (a bias artifact, not")
    print("a refutation). The estimator-free Rician-exact column is the cleaner noise-model test.")


if __name__ == "__main__":
    main()
