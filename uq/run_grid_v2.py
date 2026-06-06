"""
run_grid_v2.py
==============
Corrected full benchmark grid, rebuilt on the unified batched fitting layer
(ivim_fit.fit_batch). Classical methods fit fast in one batched call; DL methods
train once per scheme. Now produces usable numbers for ALL methods incl. DL.

Outputs:
  ivim_summary_v2.csv   -- per-cell stats (bias, dispersion) for heatmaps
  ivim_raw_v2.npz       -- raw per-voxel estimates, keyed cell -> (3, N) array
                           (rows D / Dstar / f). W1 consumes these ensembles.

Run from repo root, venv active, with ivim_simulator.py + ivim_fit.py alongside:
    python run_grid_v2.py
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

from .ivim_fit import make_model, fit_batch
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

METHODS = [
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
SCHEMES = ["clinical_sparse", "dense", "optimized"]
SNRS    = [10, 20, 40, 80]
TRUTHS  = ANCHOR_TRUTHS
N_NOISE = 200                      # accuracy/precision is stable here; lower if time-bound
SEED    = 0


def cell_stats(est, truth):
    est = est[np.isfinite(est)]
    if est.size == 0:
        return dict(median=np.nan, bias_pct=np.nan, cov_pct=np.nan,
                    iqr_rel_pct=np.nan, n_ok=0)
    med = np.median(est)
    q25, q75 = np.percentile(est, [25, 75])
    return dict(median=med, bias_pct=100 * (med - truth) / truth,
                cov_pct=100 * np.std(est) / np.mean(est) if np.mean(est) else np.nan,
                iqr_rel_pct=100 * (q75 - q25) / med if med else np.nan,
                n_ok=int(est.size))


def main():
    t0 = time.time()
    rows, raw = [], {}
    available = []
    for method in METHODS:
        # probe once
        try:
            make_model(method, B_SCHEMES[SCHEMES[0]])
            available.append(method)
        except Exception as e:
            print(f"  [skip] {method:42s} {type(e).__name__}: {str(e)[:70]}")
    print(f"\nRunning {len(available)}/{len(METHODS)} methods\n")

    for mi, method in enumerate(available, 1):
        m_t0 = time.time()
        for scheme in SCHEMES:
            bvals = B_SCHEMES[scheme]
            try:
                model = make_model(method, bvals)        # DL trains here, once per scheme
            except Exception:
                continue
            for snr in SNRS:
                for ti, truth in enumerate(TRUTHS):
                    rng = np.random.default_rng(SEED + 1000 * ti + snr)
                    sim = simulate_repeats(**truth, bvals=bvals, snr=snr,
                                           n_noise=N_NOISE, rng=rng)
                    D, Ds, F = fit_batch(model, sim["signals"])
                    raw[f"{method}__{scheme}__snr{snr}__t{ti}"] = np.stack([D, Ds, F])
                    for pname, arr, tv in (("D", D, truth["D"]),
                                           ("Dstar", Ds, truth["Dstar"]),
                                           ("f", F, truth["f"])):
                        rows.append(dict(method=method, scheme=scheme, snr=snr,
                                         truth_id=ti, param=pname, truth=tv,
                                         **cell_stats(arr, tv)))
        print(f"  ({mi}/{len(available)}) {method:42s} {time.time()-m_t0:6.1f}s")

    pd.DataFrame(rows).to_csv("ivim_summary_v2.csv", index=False)
    np.savez_compressed("ivim_raw_v2.npz", **raw)
    print(f"\nWrote ivim_summary_v2.csv ({len(rows)} rows) + ivim_raw_v2.npz "
          f"({len(raw)} cells)   [{time.time()-t0:.1f}s]")

    # headline peek: D* bias vs precision at SNR=20, clinical_sparse, truth 0
    df = pd.DataFrame(rows)
    peek = df[(df.param == "Dstar") & (df.snr == 20) &
              (df.scheme == "clinical_sparse") & (df.truth_id == 0)]
    print("\nD* @ SNR=20, clinical_sparse, truth_0  (bias vs precision):")
    for _, r in peek.sort_values("iqr_rel_pct").iterrows():
        print(f"  {r.method:42s} bias={r.bias_pct:+8.1f}%  IQR/med={r.iqr_rel_pct:8.1f}%")


if __name__ == "__main__":
    main()
