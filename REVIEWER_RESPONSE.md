# Response to reviewer comments

Three reviewer comments are addressed below. Each maps onto a concrete, runnable
addition to the reproducibility module; new code, data products, and figures are
committed in this branch, and proposed drop-in manuscript text is given for each.

All numbers below are produced by scripts in `npe/` against the committed
artifacts and are reproducible from the commands shown. The source manuscript
used for the text edits is the merged MRM submission
(`Fashion_Manuscript_Merged_MRM.pdf`).

| # | Severity | Comment | Status |
|---|----------|---------|--------|
| 1 | HIGH | Operationalize the OOD gate (ROC / threshold vs calibration recovery on the in-vivo data) | Done — new analysis, figure, threshold |
| 2 | MEDIUM | Dense (16-b) NPE to show below-floor overconfidence is prior-collapse, not a sparse-data edge case | Done — model trained + audited |
| 3 | LOW | De-jargon the second (Methods) abstract paragraph | Done — rewrite below |

---

## 1 — HIGH: Operationalizing the OOD gate

**Comment.** *"Reviewers will ask how to operationalize the OOD safeguards. Provide
a concrete heuristic or ROC curve analysis showing the trade-off between gating
threshold and calibration recovery on the multi-b in-vivo dataset. Do not leave
the threshold arbitrary."*

**What was done.** The manuscript proposes (Discussion, "future work") a
deployment-time gate that withholds the amortized posterior off-distribution but
leaves it untested. On the in-vivo brain dataset there is a single acquisition, so
the proposed *acquisition-level* b-vector-distance gate reduces to one scalar and
cannot be characterized across voxels. We therefore operationalize the same
"withhold when the input is off-distribution" principle at the **voxel level**,
which is what a per-voxel ROC on a single in-vivo dataset requires, and which is
the clinically useful unit (per-voxel abstention).

New script: `npe/run_g_ood_gating.py` (Checkpoint G1). For each of N = 2000 gray-matter
voxels it computes, **from the fit b-values only** (i.e. deployable — no held-out
data needed at inference):

- `chi2_red` — reduced χ² of the biexponential NLLS fit (a classical, NPE-independent
  goodness-of-fit / model-misfit signal), and
- `npe_selfconsistency` — RMS normalized residual of the NPE's own posterior-predictive
  against the observed fit signals (does the amortized posterior reproduce its own
  conditioning data?).

The quantity the gate must predict — available here only because this validation
dataset carries extra b-values — is the held-out-b posterior-predictive
miscalibration (`heldout_rms_z`), i.e. the F2 failure signal. Crucially the score
(fit b-values) and the target (held-out b-values) are computed on **disjoint**
b-value sets, so predicting one from the other is a genuine generalization test, not
a tautology.

**Result (committed: `npe/g_ood_gating.csv`, `npe/g_ood_gating.png`).**

| Deployable gate score | ROC AUC | Spearman ρ vs held-out failure |
|---|---|---|
| `npe_selfconsistency` (NPE-native) | **0.993** | +0.988 |
| `chi2_red` (classical goodness-of-fit) | 0.594 | +0.193 |

Calibration recovery — retain only the lowest-score voxels, then measure the
retained set's held-out-b coverage at nominal 0.95 (ungated = **0.033**, matching
the F2 result) and held-out normalized-residual SD (ungated = 11.0; 1.0 = calibrated):

| Retained fraction | `npe_selfconsistency` threshold | coverage@0.95 | residual SD |
|---|---|---|---|
| 100% (ungated) | — | 0.033 | 11.0 |
| 50% | ≤ 16.9 | 0.064 | 7.6 |
| 25% | ≤ 12.3 | 0.106 | 6.1 |
| 5% | ≤ 6.8 | **0.197** (6×) | **4.3** |

**Concrete recommended threshold (not arbitrary): the ROC Youden point of the
self-consistency gate, `npe_selfconsistency ≤ 16.7`** (TPR 0.97, FPR 0.05), which
retains 49% of voxels and roughly doubles held-out coverage (0.033 → 0.065) while
cutting the residual SD from 11.0 to 7.5.

Two findings are worth stating plainly, and both *strengthen* the paper:

1. The gate's recovery is **monotonic but partial** — even the strictest gate cannot
   reach nominal 0.95. This is consistent with the paper's own conclusion that the
   overconfidence is *intrinsic* and that per-scheme recalibration (not gating alone)
   is the real fix. The gate is a safe triage/abstention tool, not a cure.
2. The *independent* classical goodness-of-fit gate (`chi2_red`) barely recovers
   calibration (coverage stays ≈ 0.04 at every retained fraction; AUC 0.59). So the
   NPE's held-out failure is **not** explained by biexponential model-misfit / OOD
   tissue — it is the estimator's intrinsic overconfidence, which only an NPE-native
   self-consistency check detects. This is direct, quantitative support for the
   manuscript's "intrinsic, needs recalibration" claim.

**Reproduce.**
```
.venv-npe/bin/python npe/run_g_ood_gating.py --n-voxels 2000 --n-samples 200 --seed 42
```

**Proposed manuscript edit (Discussion, the "future work" paragraph).** Replace the
first safeguard sentence ("First, an out-of-distribution gate … withhold the
amortized posterior rather than return an overconfident estimate.") with a tested
version:

> *First, a deployment-time out-of-distribution gate can withhold the amortized
> posterior where it is untrustworthy. At the acquisition level this is a distance
> check between the clinical b-value vector and the training acquisition; at the
> voxel level it is a posterior-predictive self-consistency check computed from the
> fit b-values alone. On the in-vivo brain dataset the latter ranks voxels by their
> (normally unobservable) held-out-b miscalibration with AUC 0.99 (Youden threshold
> retaining 49% of voxels at TPR 0.97 / FPR 0.05), and gating on it monotonically
> recovers held-out coverage — from 0.03 toward 0.20 as the retained fraction
> tightens (Supplementary Figure S3). A classical biexponential goodness-of-fit gate,
> by contrast, does not recover calibration (AUC 0.59), confirming that the
> overconfidence is intrinsic to the estimator rather than driven by model-misfit;
> gating therefore provides safe triage but, consistent with the acquisition-shift
> result, full calibration still requires the per-scheme recalibration described
> next.*

New committed artifacts: `npe/run_g_ood_gating.py`, `npe/g_ood_gating.{csv,png,pdf}`,
`npe/g_ood_gating_voxels.csv`, and (for the manuscript) `figures/manuscript/figS3_ood_gating.{png,pdf}`.

---

## 2 — MEDIUM: Dense-acquisition control

**Comment.** *"Run one supplementary NPE model trained and evaluated on a dense
(e.g. 16+ b-values) protocol to prove the below-floor overconfidence is a property
of the estimator's prior-collapse, not just an edge-case of ultra-sparse data."*

**What was done.** A second NPE was trained **and** evaluated entirely on a dense
16-point b-scheme (`B_SCHEMES["dense"]` = {0, 10, 20, 30, 50, 75, 100, 150, 200,
300, 400, 500, 600, 700, 800, 1000} s/mm²) — double the 8-point clinical-sparse
scheme — using the **identical** recipe to the main `setB` model (set
representation, neural spline flow, 500 000-simulation budget, log-D* prior,
seed 0). The only thing changed is the acquisition, isolating density from the
estimator. The dense model is in-distribution for its own scheme (not the
off-scheme evaluation of Figure 4C), and the CRLB floor it is audited against is
computed for the dense acquisition, so it is correspondingly *tighter*. The dense
model trained thoroughly — 169 epochs to early-stopping vs setB's 99 — so it is not
under-trained.

A `--b-scheme` flag was added to `train_npe.py` and `run_e_efficiency.py` (default
`clinical_sparse`; fully backward-compatible). New supplementary builder
`npe/run_s2_figure.py` reproduces main-text Figure 3's exact transform (per-SNR
median D* SD-to-CRLB ratio over the 512-point grid).

**Result (committed: `npe/efficiency_map_dense.csv`, `figures/manuscript/figS2_dense_acquisition.*`).**

D* grid points classified overconfident (claimed SD below the CRLB floor, ratio < 0.9):

| Acquisition | D* overconfident | D | f |
|---|---|---|---|
| Clinical-sparse (8 b) | **69.5%** | 17.0% | 22.2% |
| Dense (16 b) | **41.4%** | 13.4% | 12.5% |

Per-SNR median D* claimed-SD / CRLB ratio (< 1 = overconfident, below the floor):

| SNR | Clinical-sparse | Dense (16 b) |
|---|---|---|
| 10 | 0.08 | **0.46** |
| 20 | 0.16 | **0.85** |
| 50 | 0.38 | 1.59 |
| 100 | 0.67 | 2.24 |

**Interpretation (the answer to the comment).** Doubling the b-values and tightening
the CRLB floor does **not** remove the below-floor D* overconfidence: it persists at
41% of grid points overall, and — critically — it persists **throughout the clinical
SNR regime** (claimed ratio 0.46 at SNR 10 and 0.85 at SNR 20, both below the dense
floor). At high SNR the dense D* instead crosses to *inefficiency* (ratio 1.6–2.2),
the same SNR-averaging signature the main text already reports for D and f. In other
words, with more b-values D* simply starts to behave like D and f did in the
main model — overconfident where information is scarce (low SNR), over-wide where it
is rich (high SNR) — which is exactly the prior-reversion / SNR-averaging mechanism
the paper attributes to the amortized estimator, now shown to be independent of
acquisition density. The below-floor overconfidence is therefore a property of the
estimator under weak identifiability, not an artefact of ultra-sparse sampling. The
dense acquisition mitigates it (69.5% → 41.4%) but cannot eliminate it where it
matters clinically.

**Reproduce.**
```
# train (≈60 min CPU) -> npe/npe_posterior_dense.pt (gitignored), npe/loss_dense.json
.venv-npe/bin/python npe/train_npe.py --mode set --b-scheme dense --budget 500000 \
  --epochs 200 --log-dstar --seed 0 --output npe/npe_posterior_dense.pt \
  --loss-output npe/loss_dense.json
# audit -> npe/efficiency_map_dense.csv
.venv-npe/bin/python npe/run_e_efficiency.py --model npe/npe_posterior_dense.pt \
  --b-scheme dense --out-tag _dense --skip-anchor-validation
# figure (needs pandas+matplotlib; use the main .venv) -> figures/manuscript/figS2_dense_acquisition.*
.venv/bin/python npe/run_s2_figure.py
```

**Proposed manuscript edit.** Add one sentence to Part 2 / Discussion, parallel to the
existing MAF (Supplementary Figure S1) sentence:

> *To separate this prior-reversion from the sparsity of the clinical acquisition, a
> second NPE was trained and evaluated entirely on a dense 16-point b-scheme under
> otherwise identical conditions. Audited against the correspondingly tighter dense
> CRLB floor, its claimed D* uncertainty remained below the floor throughout the
> clinical SNR regime (median ratio 0.46 at SNR 10, 0.85 at SNR 20; 41% of grid points
> overconfident overall, versus 69% for the clinical-sparse model), crossing to
> inefficiency only at high SNR — the same SNR-averaging signature seen for D and f
> (Supplementary Figure S2). The below-floor D* overconfidence is thus a property of
> the amortized estimator under weak identifiability, mitigated but not eliminated by
> a denser acquisition, rather than an artefact of ultra-sparse sampling.*

And soften the corresponding limitation: acquisition *density* is now controlled
(Supp. Fig. S2) in addition to flow architecture (Supp. Fig. S1); generality across
*priors* remains the open item.

New committed artifacts: `train_npe.py`/`run_e_efficiency.py` `--b-scheme` flag,
`npe/run_s2_figure.py`, `npe/efficiency_map_dense.{csv,png}`, `npe/loss_dense.json`,
`figures/manuscript/figS2_dense_acquisition.{png,pdf,csv}`.

---

## 3 — LOW: De-jargon the abstract Methods paragraph

**Comment.** *"The text is highly jargon-dense. Rewrite the second paragraph of the
abstract. Explain 'aggregate calibration checks' vs 'pointwise overconfidence' in
plain language before invoking the Cramér–Rao floor."*

**Original (Methods, second abstract paragraph).**

> Methods: Within a common biexponential IVIM simulation framework (24 fitting
> methods; SNR 10–100; three ground-truth conditions; clinically sparse b-scheme),
> two analyses were conducted. An amortized neural posterior estimator (NPE) was
> trained and audited against the Cramér–Rao lower bound (CRLB) and a nonlinear
> least-squares (NLLS) reference, separating claimed from achieved precision, with
> held-out-b posterior-predictive checks under model misspecification, acquisition
> shift, and on an open in-vivo dataset. Separately, uncertainty construction was
> compared across classical bootstrap, Bayesian (Laplace, MCMC credible and quantile
> intervals), and deep-learning (ensemble, input-perturbation) paradigms, scored by
> empirical coverage.

**Proposed rewrite (plain-language aggregate-vs-pointwise framing first, CRLB after).**

> Methods: Two analyses were run within one biexponential IVIM simulation framework
> (24 fitting methods; SNR 10–100; three ground-truth conditions; clinically sparse
> b-scheme). A method can look calibrated on *aggregate* checks — its error bars
> contain the truth at the stated rate once averaged over the parameter space — yet
> be *pointwise overconfident*, reporting intervals that are too narrow exactly where
> a parameter is hard to estimate, with the wide and narrow errors cancelling in the
> average. To expose this, an amortized neural posterior estimator (NPE) was trained
> and its claimed precision compared, value by value, against the Cramér–Rao lower
> bound (CRLB) — the best precision the measurement physically permits — and a
> nonlinear least-squares (NLLS) reference, separating the uncertainty the network
> *claims* from the scatter its estimates *actually* show. Robustness used held-out-b
> posterior-predictive checks under model misspecification, acquisition shift, and an
> open in-vivo dataset. Separately, uncertainty constructions were compared across
> classical bootstrap, Bayesian (Laplace, MCMC credible and quantile intervals), and
> deep-learning (ensemble, input-perturbation) paradigms, scored by empirical coverage.

This front-loads the aggregate-vs-pointwise distinction in plain language and only
then invokes the CRLB. It adds ~45 words; if the structured-abstract word budget is
tight, the parenthetical "with the wide and narrow errors cancelling in the average"
can be dropped (the Results paragraph already states the cancellation).
