"""
run_w3_calib.py
===============
W3 Checkpoint 2 — constructible-uncertainty application.

Over the headline 9-cell set (3 truths x {SNR 10, 20, 40} x clinical_sparse, N=200),
apply each method's matching per-voxel uncertainty generator and score its
calibration against the known truth:

    method                              generator(s)
    ----------------------------------  ------------------------------------------
    OGC_AmsterdamUMC_biexp              bootstrap_cell            (classical)
    OGC_AmsterdamUMC_biexp_segmented    bootstrap_cell            (classical)
    OGC_AmsterdamUMC_Bayesian_biexp     laplace_uncertainty +
                                        mcmc_uncertainty (+ its 2.5/97.5 interval)
    IVIM_NEToptim                       input_perturbation_uncertainty (predictive)
    Super_IVIM_DC                       input_perturbation_uncertainty (predictive)

Plus, for the epistemic-vs-predictive contrast figure ONLY, ensemble_uncertainty
(M=5 retrains, epistemic) on the two DL methods over a 3-cell SUBSET
(truth 0 x SNR 10/20/40 x clinical_sparse) — scoped down because the figure needs
the comparison, not exhaustive coverage.

For each (method, cell, param) we record, at nominal {0.5, 0.8, 0.9, 0.95}:
  coverage, ECE, sharpness (rel. half-width @0.95), the empirical-SD ceiling
  coverage (best-case reference), and the MCMC 2.5/97.5 quantile-interval coverage.

Output (gitignored):
    calib_w3.csv  -- long form:
        method, cell, truth_id, snr, param, generator, nominal,
        coverage, ece, sharpness, ceiling_cov, n_ok

Run from the worktree root with the project venv:
    .venv/bin/python run_w3_calib.py
    W3_SMOKE=1 .venv/bin/python run_w3_calib.py     # tiny: 1 cell, fast generators
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from . import calib
from .ivim_fit import make_model
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS
from .bootstrap import bootstrap_cell
from .bayesian import laplace_uncertainty, mcmc_uncertainty
from .dl_uncertainty import ensemble_uncertainty, input_perturbation_uncertainty

PARAMS = ("D", "Dstar", "f")
NOMINALS = [0.50, 0.80, 0.90, 0.95]
SCHEME = "clinical_sparse"
SEED = 0
N_NOISE = int(os.environ.get("N_NOISE", "200"))
SMOKE = os.environ.get("W3_SMOKE", "") == "1"
# which generator groups to run (default all); lets a fast smoke skip DL trains.
GROUPS = set((os.environ.get("W3_GROUPS")
              or "bootstrap,bayesian,perturb,ensemble").split(","))

# 9-cell headline set; subset for the DL epistemic-vs-predictive contrast.
if SMOKE:
    CELLS = [(0, 20)]
    SUBSET_CELLS = [(0, 20)]
    N_NOISE = int(os.environ.get("N_NOISE", "30"))
else:
    CELLS = [(ti, snr) for ti in (0, 1, 2) for snr in (10, 20, 40)]
    SUBSET_CELLS = [(0, snr) for snr in (10, 20, 40)]

BVALS = B_SCHEMES[SCHEME]


def cell_signals(ti, snr):
    truth = ANCHOR_TRUTHS[ti]
    rng = np.random.default_rng(SEED + 1000 * ti + snr)
    sim = simulate_repeats(**truth, bvals=BVALS, snr=snr, n_noise=N_NOISE, rng=rng)
    return sim["signals"], np.array([truth["D"], truth["Dstar"], truth["f"]], float)


def _ceiling_cov(est_col, truth_scalar, level):
    """Best-case reference: coverage if sigma were the empirical SD of the estimates."""
    fin = est_col[np.isfinite(est_col)]
    if fin.size < 2:
        return np.nan
    emp_sd = float(np.std(fin))
    if emp_sd <= 0:
        return np.nan
    return calib.coverage(est_col, truth_scalar, emp_sd, levels=[level])[round(level, 2)]


def score_rows(method, ti, snr, generator, est, sigma, truth_vec):
    """Long-form rows for one (method, cell, generator): all params x nominals."""
    cell = f"t{ti}_snr{snr}"
    out = []
    for p, pname in enumerate(PARAMS):
        est_c, sig_c, tv = est[:, p], sigma[:, p], float(truth_vec[p])
        cov = calib.coverage(est_c, tv, sig_c, levels=NOMINALS)
        ece = calib.ece(cov)
        sharp = calib.sharpness_rel(sig_c, tv, level=0.95)
        n_ok = int((np.isfinite(est_c) & np.isfinite(sig_c) & (sig_c > 0)).sum())
        for L in NOMINALS:
            out.append(dict(method=method, cell=cell, truth_id=ti, snr=snr,
                            param=pname, generator=generator, nominal=L,
                            coverage=cov[round(L, 2)], ece=ece, sharpness=sharp,
                            ceiling_cov=_ceiling_cov(est_c, tv, L), n_ok=n_ok))
    return out


def score_quantile_rows(method, ti, snr, est, lo, hi, truth_vec):
    """MCMC 2.5/97.5 credible-interval coverage (the skew-aware variant) @0.95."""
    cell = f"t{ti}_snr{snr}"
    out = []
    for p, pname in enumerate(PARAMS):
        tv = float(truth_vec[p])
        m = np.isfinite(lo[:, p]) & np.isfinite(hi[:, p])
        qcov = float(np.mean((lo[m, p] <= tv) & (tv <= hi[m, p]))) if m.any() else np.nan
        with np.errstate(invalid="ignore"):
            sharp = float(np.nanmean((hi[:, p] - lo[:, p]) / abs(tv))) if tv else np.nan
        out.append(dict(method=method, cell=cell, truth_id=ti, snr=snr,
                        param=pname, generator="mcmc_quantile", nominal=0.95,
                        coverage=qcov, ece=abs(qcov - 0.95) if np.isfinite(qcov) else np.nan,
                        sharpness=sharp, ceiling_cov=_ceiling_cov(est[:, p], tv, 0.95),
                        n_ok=int(m.sum())))
    return out


def _peek(label, rows):
    """Print a one-line D* @0.95 coverage so progress is visibly real, not just green."""
    ds = [r for r in rows if r["param"] == "Dstar" and r["nominal"] == 0.95]
    if ds:
        r = ds[0]
        print(f"      {label:24s} D*@.95 cov={r['coverage']:.3f} "
              f"ceiling={r['ceiling_cov']:.3f} sharp={r['sharpness']:.2f} n={r['n_ok']}",
              flush=True)


def main():
    t0 = time.time()
    print(f"W3 calib | N={N_NOISE} | cells={CELLS} | subset(ensemble)={SUBSET_CELLS}"
          f"{'  [SMOKE]' if SMOKE else ''}\n", flush=True)
    rows = []

    # --- classical: bootstrap -------------------------------------------------
    for method in (["OGC_AmsterdamUMC_biexp", "OGC_AmsterdamUMC_biexp_segmented"]
                   if "bootstrap" in GROUPS else []):
        print(f"[bootstrap] {method}", flush=True)
        model = make_model(method, BVALS)
        for ti, snr in CELLS:
            sig, tv = cell_signals(ti, snr)
            rng = np.random.default_rng(SEED + ti + snr)
            est, sigma = bootstrap_cell(model, sig, BVALS, K=(30 if SMOKE else 200), rng=rng)
            r = score_rows(method, ti, snr, "bootstrap", est, sigma, tv)
            rows += r; _peek(f"t{ti}/snr{snr}", r)

    # --- Bayesian: Laplace + MCMC (+ quantile interval) -----------------------
    bayes = "OGC_AmsterdamUMC_Bayesian_biexp"
    model = make_model(bayes, BVALS) if "bayesian" in GROUPS else None
    if "bayesian" in GROUPS:
        print(f"\n[laplace+mcmc] {bayes}", flush=True)
    for ti, snr in (CELLS if "bayesian" in GROUPS else []):
        sig, tv = cell_signals(ti, snr)
        est_l, sig_l = laplace_uncertainty(model, sig, BVALS)
        r = score_rows(bayes, ti, snr, "laplace", est_l, sig_l, tv)
        rows += r; _peek(f"laplace t{ti}/snr{snr}", r)
        rng = np.random.default_rng(SEED + ti + snr)
        est_m, sig_m, lo, hi = mcmc_uncertainty(
            model, sig, BVALS,
            nsteps=(400 if SMOKE else 1500), burn=(150 if SMOKE else 500), rng=rng)
        r = score_rows(bayes, ti, snr, "mcmc", est_m, sig_m, tv)
        rq = score_quantile_rows(bayes, ti, snr, est_m, lo, hi, tv)
        rows += r + rq; _peek(f"mcmc t{ti}/snr{snr}", r)

    # --- DL: input perturbation (predictive) ----------------------------------
    for method in (["IVIM_NEToptim", "Super_IVIM_DC"] if "perturb" in GROUPS else []):
        print(f"\n[input_perturbation] {method}", flush=True)
        for ti, snr in CELLS:
            sig, tv = cell_signals(ti, snr)
            rng = np.random.default_rng(SEED + ti + snr)
            est, sigma, _ = input_perturbation_uncertainty(
                method, sig, BVALS, snr=snr, B=(5 if SMOKE else 50), rng=rng, verbose=False)
            r = score_rows(method, ti, snr, "input_perturbation", est, sigma, tv)
            rows += r; _peek(f"t{ti}/snr{snr}", r)

    # --- DL: ensemble (epistemic) — SUBSET, contrast figure only --------------
    for method in (["IVIM_NEToptim", "Super_IVIM_DC"] if "ensemble" in GROUPS else []):
        print(f"\n[ensemble M=5 SUBSET] {method}", flush=True)
        for ti, snr in SUBSET_CELLS:
            sig, tv = cell_signals(ti, snr)
            est, sigma, _ = ensemble_uncertainty(
                method, sig, BVALS, M=(2 if SMOKE else 5), snr=snr, verbose=False)
            r = score_rows(method, ti, snr, "ensemble", est, sigma, tv)
            rows += r; _peek(f"t{ti}/snr{snr}", r)

    df = pd.DataFrame(rows)
    df.to_csv("calib_w3.csv", index=False)
    print(f"\nWrote calib_w3.csv ({len(df)} rows)   [{time.time()-t0:.1f}s]\n", flush=True)

    # compact stdout summary: coverage @0.95 by method x generator x param
    print("=== coverage @ nominal 0.95 (mean over cells) ===", flush=True)
    sub = df[df.nominal == 0.95]
    piv = (sub.groupby(["method", "generator", "param"])["coverage"]
              .mean().reset_index())
    cmp = (sub.groupby(["method", "generator", "param"])["ceiling_cov"]
              .mean().reset_index().rename(columns={"ceiling_cov": "ceiling"}))
    merged = piv.merge(cmp, on=["method", "generator", "param"])
    for _, g in merged.groupby(["method", "generator"]):
        m, gen = g.iloc[0]["method"], g.iloc[0]["generator"]
        cells = "  ".join(f"{row['param']}={row['coverage']:.2f}(ceil{row['ceiling']:.2f})"
                          for _, row in g.iterrows())
        print(f"  {m:34s} {gen:18s} {cells}", flush=True)

    if df["coverage"].notna().sum() == 0:
        raise SystemExit("run_w3_calib FAILED: no finite coverage numbers produced.")
    print("\nOK: produced finite coverage numbers across methods/generators.", flush=True)


if __name__ == "__main__":
    main()
