"""
run_grid_v3.py
==============
High-N ceiling grid for the IVIM uncertainty study (W3, Checkpoint 1).

Full factorial point-fit accuracy/precision grid on the unified batched fitting
layer (ivim_fit.fit_batch), at N=500 noise realizations per cell:

    methods x schemes x SNRs x truths

This is the "ceiling" pass: POINT fits only (no per-voxel sigma). It measures each
method's bias / dispersion / CoV from the spread of estimates around the known
truth, which is the empirical-SD ceiling the W3 calibration generators are judged
against.

Outputs (gitignored):
    ivim_summary_v3.csv  -- per-cell stats (median, bias%, sd, CoV%, IQR/med%, n_ok)
    ivim_raw_v3.npz      -- raw per-voxel estimates, key cell -> (3, N) [D, Dstar, f]

Hardening vs v2:
  * DL full-volume path now uses ivim_fit's chunked fallback, so a single
    off-by-one voxel can no longer NaN a whole DL cell (the 8-NaN-cell bug).
  * Every cell is checked for finite voxels; any FULLY-DEAD cell (0 finite for a
    param among an available method) aborts the run loudly at the end — no silent
    recovery. Partial per-voxel NaN (bound-pinned fits etc.) is expected and only
    reported, not fatal.

Run from the worktree root with the project venv:
    .venv/bin/python run_grid_v3.py
    N_NOISE=20 METHODS=OGC_AmsterdamUMC_biexp,IVIM_NEToptim \
        SCHEMES=clinical_sparse SNRS=20 .venv/bin/python run_grid_v3.py   # smoke
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from .ivim_fit import make_model, fit_batch
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

ALL_METHODS = [
    "OGC_AmsterdamUMC_biexp_segmented", "IAR_LU_segmented_2step",
    "IAR_LU_segmented_3step", "OJ_GU_seg", "TCML_TechnionIIT_SLS",
    "OGC_AmsterdamUMC_biexp", "IAR_LU_biexp", "PV_MUMC_biexp",
    "PvH_KB_NKI_IVIMfit", "IAR_LU_subtracted", "IAR_LU_modified_mix",
    "IAR_LU_modified_topopro", "ETP_SRI_LinearFitting", "DT_IIITN_WLS",
    "TCML_TechnionIIT_lsqlm", "TCML_TechnionIIT_lsqtrf",
    "TCML_TechnionIIT_lsqBOBYQA", "TCML_TechnionIIT_lsq_sls_lm",
    "TCML_TechnionIIT_lsq_sls_trf", "TCML_TechnionIIT_lsq_sls_BOBYQA",
    "OGC_AmsterdamUMC_Bayesian_biexp",
    "Super_IVIM_DC", "IVIM_NEToptim",
    "TF_reference_IVIMfit",
]


def _env_list(name, default):
    v = os.environ.get(name)
    return default if not v else [s.strip() for s in v.split(",") if s.strip()]


METHODS = _env_list("METHODS", ALL_METHODS)
SCHEMES = _env_list("SCHEMES", ["clinical_sparse", "dense", "optimized"])
SNRS    = [int(s) for s in _env_list("SNRS", ["10", "20", "40", "80"])]
TRUTHS  = ANCHOR_TRUTHS
N_NOISE = int(os.environ.get("N_NOISE", "500"))
SEED    = 0


def cell_stats(est, truth):
    fin = est[np.isfinite(est)]
    n_ok = int(fin.size)
    if n_ok == 0:
        return dict(median=np.nan, mean=np.nan, bias_pct=np.nan, sd=np.nan,
                    cov_pct=np.nan, iqr_rel_pct=np.nan, n_ok=0)
    med = float(np.median(fin)); mean = float(np.mean(fin)); sd = float(np.std(fin))
    q25, q75 = np.percentile(fin, [25, 75])
    return dict(
        median=med, mean=mean,
        bias_pct=100.0 * (med - truth) / truth,
        sd=sd,
        cov_pct=100.0 * sd / mean if mean else np.nan,   # coefficient of variation
        iqr_rel_pct=100.0 * (q75 - q25) / med if med else np.nan,
        n_ok=n_ok,
    )


def main():
    t0 = time.time()
    print(f"grid v3 | N={N_NOISE} | {len(METHODS)} methods x {len(SCHEMES)} schemes "
          f"x {len(SNRS)} SNR x {len(TRUTHS)} truths", flush=True)

    available = []
    for method in METHODS:
        try:
            make_model(method, B_SCHEMES[SCHEMES[0]])
            available.append(method)
        except Exception as e:
            print(f"  [skip] {method:42s} {type(e).__name__}: {str(e)[:70]}", flush=True)
    print(f"\nRunning {len(available)}/{len(METHODS)} methods\n", flush=True)

    rows, raw = [], {}
    dead_cells = []          # (method, scheme, snr, truth_id, param) with 0 finite
    for mi, method in enumerate(available, 1):
        m_t0 = time.time()
        m_min_finite = N_NOISE
        for scheme in SCHEMES:
            bvals = B_SCHEMES[scheme]
            try:
                model = make_model(method, bvals)          # DL trains here, once per scheme
            except Exception as e:
                print(f"     [build-fail] {method}/{scheme}: {type(e).__name__}", flush=True)
                continue
            for snr in SNRS:
                for ti, truth in enumerate(TRUTHS):
                    rng = np.random.default_rng(SEED + 1000 * ti + snr)
                    sim = simulate_repeats(**truth, bvals=bvals, snr=snr,
                                           n_noise=N_NOISE, rng=rng)
                    D, Ds, F = fit_batch(model, sim["signals"])
                    key = f"{method}__{scheme}__snr{snr}__t{ti}"
                    raw[key] = np.stack([D, Ds, F])
                    for pname, arr, tv in (("D", D, truth["D"]),
                                           ("Dstar", Ds, truth["Dstar"]),
                                           ("f", F, truth["f"])):
                        nfin = int(np.isfinite(arr).sum())
                        m_min_finite = min(m_min_finite, nfin)
                        if nfin == 0:
                            dead_cells.append((method, scheme, snr, ti, pname))
                        rows.append(dict(method=method, scheme=scheme, snr=snr,
                                         truth_id=ti, param=pname, truth=tv,
                                         n_total=N_NOISE, **cell_stats(arr, tv)))
        flag = "  <-- has dead cell(s)!" if m_min_finite == 0 else ""
        print(f"  ({mi}/{len(available)}) {method:42s} {time.time()-m_t0:7.1f}s  "
              f"min_finite/cell={m_min_finite}/{N_NOISE}{flag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("ivim_summary_v3.csv", index=False)
    np.savez_compressed("ivim_raw_v3.npz", **raw)
    print(f"\nWrote ivim_summary_v3.csv ({len(df)} rows) + ivim_raw_v3.npz "
          f"({len(raw)} cells)   [{time.time()-t0:.1f}s]", flush=True)

    # positive-output verification
    print("\n=== finite-voxel verification ===", flush=True)
    per_method = (df.assign(frac=df.n_ok / df.n_total)
                    .groupby("method")["frac"].agg(["min", "mean"]))
    for m, r in per_method.iterrows():
        print(f"  {m:42s} finite frac  min={r['min']:.3f}  mean={r['mean']:.3f}",
              flush=True)

    if dead_cells:
        print(f"\n!!! {len(dead_cells)} FULLY-DEAD cell(s) (0 finite voxels) !!!", flush=True)
        for c in dead_cells[:40]:
            print(f"    DEAD {c[0]} / {c[1]} / snr{c[2]} / t{c[3]} / {c[4]}", flush=True)
        raise SystemExit(
            f"run_grid_v3 FAILED: {len(dead_cells)} dead cells — the DL NaN bug "
            f"(or a new failure) is NOT resolved. No silent recovery."
        )

    print("\nOK: zero fully-dead cells. Every available method produced finite "
          "estimates in every cell.", flush=True)


if __name__ == "__main__":
    main()
