"""
run_p3_grid.py
==============
Full factorial benchmark grid for the OSIPI IVIM study.

For every (method × b-value scheme × SNR × ground-truth), fit N_NOISE noisy
realizations through OsipiBase and record bias + precision for D, D*, f.
Writes a tidy long-format CSV that P4 turns into heatmaps.

ROBUSTNESS: each algorithm is probed once; if it fails to initialize (e.g.
needs MATLAB, or a DL backend that isn't set up) it is SKIPPED and logged,
so the grid always completes. A per-fit try/except records failures per cell.

RUN FROM REPO ROOT, venv active, with ivim_simulator.py in the same root:

    # 1) smoke test first  -> set N_NOISE = 25, confirm the roster looks right
    # 2) real run          -> set N_NOISE = 250+, let it run
    python run_p3_grid.py
"""

from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

from src.wrappers.OsipiBase import OsipiBase
from .ivim_simulator import simulate_repeats, B_SCHEMES, ANCHOR_TRUTHS

# --- grid config ----------------------------------------------------------
METHODS = [
    # segmented LSQ
    "OGC_AmsterdamUMC_biexp_segmented", "IAR_LU_segmented_2step",
    "IAR_LU_segmented_3step", "OJ_GU_seg", "TCML_TechnionIIT_SLS",
    # full / simultaneous LSQ
    "OGC_AmsterdamUMC_biexp", "IAR_LU_biexp", "PV_MUMC_biexp",
    "PvH_KB_NKI_IVIMfit", "IAR_LU_subtracted", "IAR_LU_modified_mix",
    "IAR_LU_modified_topopro", "ETP_SRI_LinearFitting", "DT_IIITN_WLS",
    "TCML_TechnionIIT_lsqlm", "TCML_TechnionIIT_lsqtrf",
    "TCML_TechnionIIT_lsqBOBYQA", "TCML_TechnionIIT_lsq_sls_lm",
    "TCML_TechnionIIT_lsq_sls_trf", "TCML_TechnionIIT_lsq_sls_BOBYQA",
    # Bayesian
    "OGC_AmsterdamUMC_Bayesian_biexp",
    # deep learning
    "Super_IVIM_DC", "IVIM_NEToptim",
    # institution-specific + reference
    "ASD_MemorialSloanKettering_QAMPER_IVIM", "TF_reference_IVIMfit",
]
SCHEMES = ["clinical_sparse", "dense", "optimized"]
SNRS    = [10, 20, 40, 80]
TRUTHS  = ANCHOR_TRUTHS                  # 3 anchor points
N_NOISE = 25                             # <-- 25 for smoke test, then 250+
SEED    = 0
OUT_CSV = "ivim_benchmark_results.csv"


def parse_result(res):
    if isinstance(res, dict):
        D  = res.get("D")
        Ds = res.get("Dp", res.get("D*", res.get("Dstar", res.get("Ds"))))
        f  = res.get("f", res.get("Fp"))
        return D, Ds, f
    raise TypeError(f"Unexpected fit return: {type(res)} {res!r}")


def probe_methods(bvals):
    """Try to construct each algorithm once; return {name: model|None} + log."""
    roster, skipped = {}, {}
    for name in METHODS:
        try:
            roster[name] = OsipiBase(bvalues=bvals, algorithm=name)
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {str(e)[:80]}"
    return roster, skipped


def cell_stats(est, truth):
    est = est[np.isfinite(est)]
    n = est.size
    if n == 0:
        return dict(median=np.nan, mean=np.nan, bias_pct=np.nan,
                    cov_pct=np.nan, iqr_rel_pct=np.nan, n_ok=0)
    med = np.median(est)
    q25, q75 = np.percentile(est, [25, 75])
    return dict(
        median=med, mean=np.mean(est),
        bias_pct=100 * (med - truth) / truth,
        cov_pct=100 * np.std(est) / np.mean(est) if np.mean(est) else np.nan,
        iqr_rel_pct=100 * (q75 - q25) / med if med else np.nan,   # robust dispersion
        n_ok=n,
    )


def main():
    t0 = time.time()
    bref = B_SCHEMES[SCHEMES[0]]
    roster, skipped = probe_methods(bref)

    print(f"Methods available: {len(roster)} / {len(METHODS)}")
    for n in roster: print(f"  [run]  {n}")
    for n, why in skipped.items(): print(f"  [skip] {n:42s} {why}")
    print(f"\nGrid: {len(roster)} methods x {len(SCHEMES)} schemes x "
          f"{len(SNRS)} SNR x {len(TRUTHS)} truths x {N_NOISE} noise\n")

    rows = []
    for mi, method in enumerate(roster, 1):
        m_t0 = time.time()
        for scheme in SCHEMES:
            bvals = B_SCHEMES[scheme]
            try:
                model = OsipiBase(bvalues=bvals, algorithm=method)
            except Exception:
                continue
            for snr in SNRS:
                for ti, truth in enumerate(TRUTHS):
                    rng = np.random.default_rng(SEED + 1000 * ti + snr)
                    sim = simulate_repeats(**truth, bvals=bvals, snr=snr,
                                           n_noise=N_NOISE, rng=rng)
                    D, Ds, F, fails = [], [], [], 0
                    for sig in sim["signals"]:
                        try:
                            d, ds, f = parse_result(model.osipi_fit(sig, bvals))
                            D.append(d); Ds.append(ds); F.append(f)
                        except Exception:
                            fails += 1
                    for pname, arr, tv in (("D", D, truth["D"]),
                                           ("Dstar", Ds, truth["Dstar"]),
                                           ("f", F, truth["f"])):
                        s = cell_stats(np.array(arr, float), tv)
                        rows.append(dict(method=method, scheme=scheme, snr=snr,
                                         truth_id=ti, param=pname, truth=tv,
                                         n_fail=fails, **s))
        print(f"  ({mi}/{len(roster)}) {method:42s} {time.time()-m_t0:6.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}   ({time.time()-t0:.1f}s total)")

    # quick console peek: D* dispersion by method at SNR=20, clinical_sparse
    peek = df[(df.param == "Dstar") & (df.snr == 20) &
              (df.scheme == "clinical_sparse")]
    if not peek.empty:
        print("\nD* robust dispersion (IQR/median %) @ SNR=20, clinical_sparse:")
        for _, r in peek.sort_values("iqr_rel_pct").iterrows():
            print(f"  {r.method:42s} {r.iqr_rel_pct:7.1f}%  bias={r.bias_pct:+6.1f}%")


if __name__ == "__main__":
    main()
