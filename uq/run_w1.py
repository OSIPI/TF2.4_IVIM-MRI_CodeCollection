"""
run_w1.py
=========
Apply the calibration ruler to the raw ensembles from run_grid_v2 (ivim_raw_v2.npz).

Phase A construction: use each method's empirical ensemble SD as its best-case
reported sigma, then measure coverage of truth. This already yields the headline
— a method that is precise but biased (tight sigma, wrong center) under-covers
badly, quantifying overconfidence — without yet needing W2's per-voxel uncertainty.

Run from repo root with calib.py + ivim_simulator.py alongside and ivim_raw_v2.npz present:
    python run_w1.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .calib import coverage, ece, sharpness_rel
from .ivim_simulator import ANCHOR_TRUTHS

PARAMS = ["D", "Dstar", "f"]

def main():
    raw = np.load("ivim_raw_v2.npz")
    rows = []
    for key in raw.files:
        method, scheme, snrtag, ttag = key.split("__")
        snr = int(snrtag[3:]); ti = int(ttag[1:])
        truth = ANCHOR_TRUTHS[ti]
        arr = raw[key]                              # (3, N): D, Dstar, f
        for pi, pname in enumerate(PARAMS):
            est = arr[pi]
            fin = est[np.isfinite(est)]
            if fin.size < 10:
                continue
            tv = truth[pname]
            sigma = np.std(fin)                     # best-case reported sigma (Phase A)
            cov = coverage(est, tv, sigma)
            rows.append(dict(
                method=method, scheme=scheme, snr=snr, truth_id=ti, param=pname,
                truth=tv, median=np.median(fin),
                bias_pct=100 * (np.median(fin) - tv) / tv,
                sigma=sigma, cov95=cov[0.95], cov68=cov[0.68],
                ece=ece(cov), sharp_rel=sharpness_rel(sigma, tv), n=fin.size))

    df = pd.DataFrame(rows)
    df.to_csv("calib_w1.csv", index=False)
    print(f"Wrote calib_w1.csv ({len(df)} rows)\n")

    # headline: D* calibration at SNR=20, clinical_sparse, truth_0 — ranked by ECE
    peek = df[(df.param == "Dstar") & (df.snr == 20) &
              (df.scheme == "clinical_sparse") & (df.truth_id == 0)]
    print("D* calibration @ SNR=20, clinical_sparse, truth_0  (ranked by ECE):")
    print(f"  {'method':42s} {'bias%':>8} {'cov95':>7} {'ECE':>6} {'sharp':>7}")
    for _, r in peek.sort_values("ece").iterrows():
        print(f"  {r.method:42s} {r.bias_pct:+8.1f} {r.cov95:7.2f} "
              f"{r.ece:6.3f} {r.sharp_rel:7.1f}")
    print("\nRead: cov95 near 0 with low sigma = confidently wrong (overconfident).")
    print("High cov95 with huge sharp = honest but uselessly wide (underconfident).")

if __name__ == "__main__":
    main()
