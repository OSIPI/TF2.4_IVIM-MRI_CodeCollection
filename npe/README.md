# Phase E — NPE scaffold (CP1)

Scaffolding for an **amortized neural posterior estimator (NPE)** over the IVIM
biexponential model, built on [`sbi`](https://github.com/sbi-dev/sbi). This
checkpoint delivers the simulator, prior, a smoke test, and a `pytest` suite.
**No network is trained here** — that is the next step. The only compute is the
~1000-sample smoke test.

θ = `[D, Dstar, f]` in **absolute units** (mm²/s, mm²/s, unitless).

## Why a separate environment

`sbi` requires `numpy ≥ 2` / `scipy ≥ 1.13`, which conflicts with the OSIPI/DL
stack pinned in the repo's `.venv` (Python 3.9, numpy 1.x). So the NPE work lives
in its own isolated venv at `~/ProjectFashion/.venv-npe` and this directory is
self-contained: it carries its **own copy** of `ivim_simulator.py` (pure numpy),
so the new venv needs only `numpy + scipy + torch + sbi` — never the OSIPI/DL
packages. `ivim_simulator.py` here is a verbatim copy of `uq/ivim_simulator.py`.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv ~/ProjectFashion/.venv-npe          # Python 3.10+
~/ProjectFashion/.venv-npe/bin/pip install -r npe/requirements.txt
```

Verified resolved versions: **sbi 0.26.1, torch 2.12.0, numpy 2.4.6,
scipy 1.17.1** (CPython 3.12.13). The venv is git-ignored.

## Run

```bash
# smoke test (≈1000 draws, all modes, embedding, timing)
cd npe && PYTHONPATH=. ~/ProjectFashion/.venv-npe/bin/python run_e_smoke.py

# unit tests
cd npe && ~/ProjectFashion/.venv-npe/bin/python -m pytest
```

`PYTHONPATH=.` (smoke) and `conftest.py` (pytest) put this flat directory on the
import path; it is intentionally not a Python package.

## Files

| File | Role |
|------|------|
| `ivim_simulator.py` | Pure-numpy forward model (copy of `uq/ivim_simulator.py`). |
| `npe_prior.py` | `BoxUniform` prior over `[D, Dstar, f]`; `process_prior`; display scaling. |
| `npe_simulator.py` | Batched `theta → (observation, snr_context)` simulator; set-embedding helper. |
| `run_e_smoke.py` | No-training smoke check (Step 3). |
| `tests/test_npe_scaffold.py` | Invariants (Step 4). |

## Design (settled by the CP0 audit)

- **Prior** — `BoxUniform`, absolute units:
  `D ∈ [0.2e-3, 3.0e-3]`, `Dstar ∈ [3.0e-3, 0.15]`, `f ∈ [0.0, 0.5]`.
  Deliberately spans `f → 0` (mono-exp) and `Dstar → D` (degeneracy) so the
  posterior must represent that uncertainty. The prior is the **sole gatekeeper**
  of ranges — the forward model does no clamping.
- **Display scaling** — everything is absolute mm²/s internally; `D, Dstar` are
  multiplied by **×1000** (→ 1e-3 mm²/s ≡ µm²/ms) **only** at the reporting
  boundary (`npe_prior.to_display`). `f` is never scaled.
- **SNR is known context, not a θ dimension** — a scalar SNR is drawn per
  simulation (log-uniform over a configurable range) and emitted as
  `snr_context` = `log10(SNR)`, shape `(B, 1)`. The atlas is SNR-resolved by
  design.
- **Two flag-selectable axes:**
  - `clean` vs Rician-noisy — identical except the noise term (`clean` for later
    calibration ground truth; noisy for later training).
  - representation `"masked_grid"` (fixed-length `[signal | mask]` on a superset
    b-grid → simple-MLP path) vs `"set"` (`(b_i, S_i)` pairs →
    `PermutationInvariantEmbedding`, the principled path for heterogeneous
    external b-grids).

## Measured cost

The smoke test reports per-simulation cost (~1 µs/sim single-process, so 1e6
training simulations ≈ 1 s) so the training-simulation budget can be sized next.
Re-run `run_e_smoke.py` to measure on your machine.
