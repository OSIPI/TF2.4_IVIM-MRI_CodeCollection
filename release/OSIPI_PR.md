# OSIPI PR — IVIM uncertainty-calibration & CRLB efficiency benchmark module

**PR title:** Add IVIM uncertainty-calibration & CRLB efficiency benchmark (analysis module)

## Summary

This PR contributes a self-contained **benchmark/analysis module** to the OSIPI
TF2.4 IVIM-MRI Code Collection. It does not add a new fit algorithm; it adds
reproducible analysis that builds on the existing, **unmodified** fitters under
`src/`:

- a **cross-paradigm uncertainty-calibration benchmark** (`uq/`) that constructs
  per-voxel uncertainty for the classical, Bayesian, and deep-learning fitting
  paradigms and scores it with one ruler — coverage(L), expected calibration
  error (ECE), and sharpness — over a grid of truths × SNRs × b-schemes;
- a **CRLB efficiency audit** (`npe/`) of an independent `sbi`-based amortized
  neural posterior estimator, comparing its claimed precision to the empirical
  precision and to an NLLS baseline against the Cramér–Rao lower bound, plus
  held-out-b robustness checks (misspecification and a real-data demo).

It ships a reproducible calibration grid and efficiency map with committed data
products (`npe/efficiency_map.csv`, `npe/f1_misspecification.csv`,
`npe/f2_realdata.csv`, `figures/manuscript/regime_fractions.csv`) and the
manuscript figures.

## Contributors

- Avery Karlin — Independent Researcher — averykarlin3@gmail.com (sole author).

## Relationship to the OSIPI contribution template

The OSIPI guidelines (`doc/guidelines_for_contributions.md`) describe a
**per-fit-algorithm** template: code under `src/original/Initials_Institution/`,
a wrapper harmonizing to `OsipiBase` in `src/wrappers/`, a corresponding test
file, and a row in `doc/code_contributions_record.csv`. This contribution is a
benchmark/analysis module that **reuses** the unmodified upstream fitters and
adds an independent NPE — it introduces **no new `OsipiBase` fit algorithm** — so
it is contributed as an analysis module alongside (not inside) the per-algorithm
tree, referencing `src/` rather than extending it.

| Template field | Applies? | Notes |
|---|---|---|
| `src/original/Initials_Institution/` algorithm folder | **N/A** | No new fit algorithm; the module wraps existing `src/` fitters. |
| `src/wrappers/` `OsipiBase` wrapper | **N/A** | No new fit method to harmonize; `uq/` consumes the existing `OsipiBase` interface. |
| Row in `doc/code_contributions_record.csv` | **N/A** | That record enumerates fit algorithms; this is an analysis/benchmark module. |
| Test file | **Yes** | Analysis-layer tests under `uq/tests/` and `npe/tests/` (isolated from the upstream test tree). |
| Contributor names + affiliations in PR | **Yes** | Listed above. |
| Brief statement of purpose | **Yes** | See Summary. |
| Reproduction instructions | **Yes** | See `release/README.md` reproduction table. |

**Reason for deviation:** forcing this into the per-algorithm template would
require fabricating an algorithm folder and `code_contributions_record.csv` entry
for code that is not a fitter. The benchmark-module form keeps the upstream
algorithm registry accurate while still being fully reproducible.

## Contributor checklist

- [x] Contributor names and affiliations listed in the PR message.
- [x] PR message states the purpose of the contribution.
- [x] Upstream fitting code under `src/` is unmodified.
- [x] Additive change on a dedicated branch; one reviewable PR; no auto-merge.
- [x] Tests included for the new analysis code, isolated from the upstream test
      suite: `uq/tests/` (6 tests — `.venv/bin/python -m pytest uq`, or
      `make test`) and `npe/tests/` (14 tests —
      `PYTHONPATH=npe .venv-npe/bin/python -m pytest npe/tests`).
- [x] Reproduction documented with exact commands and output files
      (`release/README.md`).
- [x] Environments documented (main `.venv` and isolated `.venv-npe`) with pinned
      versions.
- [x] Data provenance cited (OSIPI TF2.4 open data, Zenodo record 14605039).
- [x] License confirmed (Apache-2.0, inherited from upstream).
- [ ] `code_contributions_record.csv` row — intentionally omitted (N/A; not a fit
      algorithm).
- [ ] `OsipiBase` wrapper — intentionally omitted (N/A; no new fit algorithm).
- [ ] Publication DOI — pending; manuscript in submission to MRM (placeholder in
      `CITATION.cff` / `release/README.md`, to update on acceptance).

## Links

- Related manuscript: **in submission to *Magnetic Resonance in Medicine*
  (MRM)**; no preprint. Citation is a placeholder to be updated on acceptance
  (see `release/README.md`).
- Module README and full reproduction table: `release/README.md`.
- Reproduction entry points:
  - 24-method accuracy/precision grid — `python -m uq.run_grid_v3` (Table 1).
  - Calibration campaign — `python -m uq.run_w3_calib` → `calib_w3.csv`
    (Table 1; Figures 1–2 source).
  - NPE training — `npe/train_npe.py --mode set` → `npe/npe_posterior_setB.pt`.
  - Efficiency map — `npe/run_e_efficiency.py` → `npe/efficiency_map.csv`
    (Table 2; Figure 3 source).
  - Robustness — `npe/run_f_robustness.py` (Figure 4, F1) and
    `npe/run_f_realdata.py` (Figure 4, F2).
  - Manuscript figures — `python make_manuscript_figures.py` (Figures 1–3 +
    `regime_fractions.csv`).
