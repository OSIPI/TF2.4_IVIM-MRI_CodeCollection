# IVIM Uncertainty Calibration & Efficiency Benchmark — Reproducibility Module

This module is a cross-paradigm **uncertainty-calibration benchmark** for
intravoxel incoherent motion (IVIM) diffusion-MRI fitting, plus a **Cramér–Rao
lower bound (CRLB) efficiency audit** of an amortized neural posterior estimator
(NPE). It is built on top of the **OSIPI TF2.4 IVIM-MRI Code Collection**: the
upstream fitting engines under `src/` are unmodified, and this module adds two
original analysis layers — `uq/` (which constructs per-voxel uncertainty for each
fitting paradigm and scores it with a single calibration ruler: coverage, ECE,
sharpness) and `npe/` (an independent `sbi`-based amortized posterior whose
reported precision is audited against the CRLB floor and an NLLS baseline, plus
held-out-b robustness checks). It is a benchmark/analysis module, not a new fit
algorithm.

## Repository structure (reproducibility-relevant files)

```
.
├── src/                              # upstream OSIPI fitters (UNMODIFIED)
│   └── wrappers/OsipiBase.py         # harmonized fit interface reused by uq/
├── uq/                               # calibration analysis layer (original)
│   ├── ivim_simulator.py            # Rician-noise IVIM forward model / ground truth
│   ├── ivim_fit.py                  # unified batched fitting wrapper over src/
│   ├── bootstrap.py                 # residual bootstrap (classical) uncertainty
│   ├── bayesian.py                  # Laplace + MCMC (emcee) posterior uncertainty
│   ├── dl_uncertainty.py            # deep ensemble + Rician input-perturbation
│   ├── calib.py                     # calibration ruler: coverage(L), ECE, sharpness
│   ├── run_grid_v3.py               # 24-method accuracy/precision grid (Table 1)
│   ├── run_w3_calib.py              # calibration campaign -> calib_w3.csv (Table 1, Figs 1-2)
│   └── make_figures.py             # interactive/headline reliability + heatmap views
├── npe/                              # amortized posterior + efficiency audit (original)
│   ├── ivim_simulator.py            # verbatim copy of uq/ivim_simulator.py (numpy-only)
│   ├── npe_prior.py                 # BoxUniform prior over [D, Dstar, f]
│   ├── npe_simulator.py             # theta -> (observation, snr_context) simulator
│   ├── train_npe.py                 # trains the NPE posterior (.pt, gitignored)
│   ├── run_cp3_validation.py        # SBC / coverage / CRLB+NLLS helpers
│   ├── run_cp4_atlas.py             # identifiability atlas (supporting)
│   ├── run_e_efficiency.py          # efficiency map -> efficiency_map.csv (Table 2, Fig 3)
│   ├── run_f_robustness.py          # F1 misspecification -> f1_misspecification.csv (Fig 4)
│   ├── run_f_realdata.py            # F2 real-data demo -> f2_realdata.csv (Fig 4)
│   ├── efficiency_map.csv           # committed data product (Table 2 / Fig 3)
│   ├── f1_misspecification.csv      # committed data product (Fig 4, F1)
│   ├── f2_realdata.csv              # committed data product (Fig 4, F2)
│   └── requirements.txt             # isolated .venv-npe pins
├── make_manuscript_figures.py       # builds Figs 1-4 + regime_fractions.csv
├── figures/manuscript/              # committed figures + regime_fractions.csv
├── utilities/data_simulation/Download_data.py   # fetches OSIPI TF2.4 data (Zenodo)
├── Makefile                         # one-command reproduction targets
├── requirements.txt                 # main .venv pins (OSIPI/DL stack)
└── release/                         # this module's documentation
```

## Environment setup

Two interpreters are required; they are deliberately separate because of a
**Python-version conflict**. The NPE stack (`sbi==0.26.1`, `torch==2.12.0`) needs
**Python ≥ 3.10** (both declare `Requires-Python: >=3.10`), whereas the main
OSIPI/DL fitting stack runs on **Python 3.9** — so `sbi` cannot be installed into
the main environment at all. Both environments otherwise run NumPy 2.x; the split
is about the interpreter version, not a NumPy major-version conflict.

### 1. Main environment `.venv` — calibration grid + 24-method fitting

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Targets the OSIPI/DL stack on **Python 3.9**. The packages in `requirements.txt`
are unpinned; the currently resolved versions in this environment are
**`numpy==2.0.2`, `scipy==1.13.1`, `torch==2.8.0`**. Key packages:
`super-ivim-dc>1.0.0`, `torch`, `ivimnet`, `dipy`, `nlopt`, `nipype`, `scipy`,
`numpy`, `matplotlib`, `pandas`. (This stack does **not** install `sbi`, which
requires Python ≥ 3.10 — see `.venv-npe` below.)

### 2. Isolated environment `.venv-npe` — NPE / efficiency map / robustness

```bash
python3.12 -m venv .venv-npe
.venv-npe/bin/pip install -r npe/requirements.txt
```

Verified resolved versions (from `npe/requirements.txt`, CPython 3.12.13):
**`sbi==0.26.1`, `torch==2.12.0`, `numpy==2.4.6`, `scipy==1.17.1`**. The `npe/`
directory carries its own numpy-only copy of `ivim_simulator.py`, so this venv
never imports the OSIPI/DL stack. All `npe/` commands are run **from the repo
root** (consistent with the reproduction table below): the `run_*` scripts add
their own directory to `sys.path`, so they import the local modules regardless of
working directory, while `train_npe.py` is invoked with `PYTHONPATH=npe`. The test
suite uses `npe/conftest.py` for imports.

### Figures only

Regenerating figures from existing CSVs needs only **`matplotlib` + `pandas`**;
either environment satisfies this.

## Reproduction table

Each paper Table/Figure, the exact command that regenerates it, and the output
file. Run from the repo root unless noted. Artifacts marked **(regenerated)** are
gitignored and not committed — the command regenerates them, and any prerequisite
is listed. Artifacts marked **(committed)** are present in the repo.

| Paper artifact | Environment | Command | Output | Status |
|---|---|---|---|---|
| **Table 1** — 24-method accuracy/precision grid | `.venv` | `make grid` (`python -m uq.run_grid_v3`) | `ivim_summary_v3.csv`, `ivim_raw_v3.npz` | regenerated |
| **Table 1** — calibration coverage / ECE / sharpness | `.venv` | `make calib` (`python -m uq.run_w3_calib`) | `calib_w3.csv` | regenerated |
| **Figure 1** — reliability diagrams | `.venv` (or matplotlib+pandas) | `python make_manuscript_figures.py` (needs `calib_w3.csv`) | `figures/manuscript/fig1_reliability_diagrams.{png,pdf}` | committed |
| **Figure 2** — calibration heatmap | `.venv` (or matplotlib+pandas) | `python make_manuscript_figures.py` (needs `calib_w3.csv`) | `figures/manuscript/fig2_calibration_heatmap.{png,pdf}` | committed |
| (NPE prerequisite) — train posterior | `.venv-npe` | `PYTHONPATH=npe .venv-npe/bin/python npe/train_npe.py --mode set --output npe/npe_posterior_setB.pt --loss-output npe/loss_setB.json` | `npe/npe_posterior_setB.pt`, `npe/loss_setB.json` | model regenerated; loss committed |
| **Table 2** — efficiency map (claimed / empirical / NLLS vs CRLB) | `.venv-npe` | `.venv-npe/bin/python npe/run_e_efficiency.py` | `npe/efficiency_map.csv`, `npe/efficiency_map.png` | committed |
| **Figure 3** — efficiency audit | matplotlib+pandas | `python make_manuscript_figures.py` (needs `calib_w3.csv` + `npe/efficiency_map.csv`) | `figures/manuscript/fig3_efficiency_audit.{png,pdf}` | committed |
| **F1 data** — held-out-b misspecification (Figure 4 input) | `.venv-npe` | `.venv-npe/bin/python npe/run_f_robustness.py` (needs `npe/npe_posterior_setB.pt`) | `npe/f1_misspecification.csv`, `npe/f1_misspecification.png` | committed |
| **F2 data** — real-data overconfidence demo (Figure 4 input) | `.venv-npe` | `.venv-npe/bin/python npe/run_f_realdata.py` (needs `npe/npe_posterior_setB.pt` + `download/Data/brain.*`) | `npe/f2_realdata.csv`, `npe/f2_realdata.png` | committed |
| `regime_fractions.csv` (manuscript SI) | matplotlib+pandas | `python make_manuscript_figures.py` | `figures/manuscript/regime_fractions.csv` | committed |
| **Figure 4** — robustness + real-data | matplotlib+pandas | `python make_manuscript_figures.py` (needs `npe/f1_misspecification.csv`, `npe/f2_realdata.csv`, `figures/manuscript/regime_fractions.csv`) | `figures/manuscript/fig4_robustness.{png,pdf}` | committed |

Notes:

- **`make_manuscript_figures.py` builds all four figures** (Figures 1–4) plus
  `regime_fractions.csv` from the committed CSVs, using only `matplotlib` +
  `pandas`. Figure 4 is the combined three-panel `fig4_robustness.{png,pdf}`
  (held-out-b coverage on simulated and real data, plus the acquisition-shift
  regime comparison); the per-panel source artifacts `npe/f1_misspecification.png`
  and `npe/f2_realdata.png` remain committed as the upstream F1/F2 outputs.
- The trained model `npe/npe_posterior_setB.pt` is **gitignored** (`*.pt`); the
  efficiency/robustness/real-data scripts default to that path, so train it first.
- `calib_w3.csv`, `ivim_summary_v3.csv`, and the `*.npz` grid artifacts are
  **gitignored**; the committed Figures 1–4 are the verifiable outputs.
- A fast non-DL check is available: `make smoke`.

## Data provenance

- **OSIPI TF2.4 open data** — Zenodo record **14605039**
  (`OSIPI_TF24_data_phantoms.zip`). Fetched by
  `utilities/data_simulation/Download_data.py` (via `zenodo_get`) into
  `download/Data/` (gitignored): <https://zenodo.org/records/14605039>.
- **F2 multi-b dataset** — the in-vivo **brain** acquisition from the same OSIPI
  TF2.4 Zenodo record: `download/Data/brain.nii.gz`, `brain.bval`,
  `brain_mask_gray_matter.nii.gz` (referenced in `npe/run_f_realdata.py`). Per
  `download/Data/brain_readme.txt` it was *"Scanned on a Philips 3T system"* (15
  unique b-values, 0–1000 s/mm²) — a real acquisition, not a digital phantom, and
  not a separate external dataset.
- **Note on the archive name.** Despite being named `..._phantoms.zip`, the Zenodo
  record bundles two distinct data types: digital phantoms under
  `download/Phantoms/` (e.g. the ICBM-atlas brain phantom and the XCAT set) and
  real in-vivo scans under `download/Data/`. The F2 demo uses the real
  `download/Data/brain` acquisition, **not** the `download/Phantoms/brain` digital
  phantom.

## License

Apache-2.0 (see `LICENSE`), inherited from the upstream OSIPI TF2.4 IVIM-MRI Code
Collection. This module is additive and released under the same license.

## Citation

The related publication is **in submission to *Magnetic Resonance in Medicine*
(MRM)**; there is no preprint. The block below is a placeholder to be updated on
acceptance.

```bibtex
@article{karlin_ivim_uncertainty_INSUBMISSION,
  author  = {Karlin, Avery},
  title   = {{Calibration and Efficiency of Uncertainty Estimates in Intravoxel Incoherent Motion Imaging: Quantile Intervals, Cross-Paradigm Comparison, and a Cram\'er--Rao Audit of Amortized Posteriors}},
  journal = {in submission to MRM},
  year    = {2026}
}
```

`CITATION.cff` resolves to this manuscript (GitHub's "Cite this repository").
Please **also** cite the upstream OSIPI TF2.4 IVIM-MRI Code Collection (see the
"Citing" section of `README_upstream.md`) when using the fitting engines under
`src/`.

## Contact

Avery Karlin — Independent Researcher — <averykarlin3@gmail.com>
