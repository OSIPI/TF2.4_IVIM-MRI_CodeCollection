"""
ivim_simulator.py
=================
Ground-truth IVIM signal simulator for the OSIPI TF2.4 fitting benchmark.

Generates diffusion-weighted signals from the biexponential IVIM model with a
*known* (D, D*, f), then corrupts them with Rician noise at a target SNR. Because
the truth is known, any deviation a fitting method produces is measurable bias or
imprecision rather than guesswork.

UNITS — aligned with OsipiBase so output plugs straight into the OSIPI fits:
    D, D*  : mm^2/s        (pancreas-typical: D ~1.1e-3, D* ~3e-2)
    f      : unitless      (perfusion fraction, 0..1)
    b      : s/mm^2
    S0     : normalized to 1.0 by default (most OSIPI fits assume normalized signal)

Output array shapes also match the toolbox convention:
    signals -> (n_voxels, n_bvalues)
    bvalues -> (n_bvalues,)
so a later fit is just:  OsipiBase(algorithm=...).osipi_fit(signals, bvalues)

No torch / no GPU. numpy only.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Physiologically plausible pancreas priors (literature-typical ranges).
# Used for random sampling; for a clean bias/precision study you normally FIX
# the truth at representative points and repeat over noise (see simulate_repeats).
# ---------------------------------------------------------------------------
PANCREAS_PRIORS = {
    "D":     (0.8e-3, 1.6e-3),   # true (tissue) diffusion, mm^2/s
    "Dstar": (1.0e-2, 8.0e-2),   # pseudo-diffusion, mm^2/s
    "f":     (0.10,   0.30),     # perfusion fraction
}

# Representative single-point truths to anchor the benchmark grid.
ANCHOR_TRUTHS = [
    {"D": 1.1e-3, "Dstar": 3.0e-2, "f": 0.15},   # typical pancreatic parenchyma
    {"D": 1.3e-3, "Dstar": 5.0e-2, "f": 0.25},   # higher perfusion
    {"D": 0.9e-3, "Dstar": 1.5e-2, "f": 0.10},   # low-f, hard regime
]

# b-value sampling schemes (s/mm^2).
B_SCHEMES = {
    "clinical_sparse": np.array([0, 50, 100, 200, 400, 600, 800, 1000], float),
    "dense":           np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 300,
                                 400, 500, 600, 700, 800, 1000], float),
    # placeholder for an optimization-derived scheme; refine in P3.
    "optimized":       np.array([0, 10, 30, 60, 150, 300, 500, 800], float),
}


# ---------------------------------------------------------------------------
# Forward model + noise
# ---------------------------------------------------------------------------
def ivim_signal(bvals, D, Dstar, f, S0=1.0):
    """Noise-free IVIM biexponential signal.

    S(b)/S0 = f * exp(-b * D*) + (1 - f) * exp(-b * D)
    """
    bvals = np.asarray(bvals, dtype=float)
    return S0 * (f * np.exp(-bvals * Dstar) + (1.0 - f) * np.exp(-bvals * D))


def add_rician_noise(signal, snr, S0=1.0, rng=None):
    """Add Rician noise. SNR is defined on S0 (i.e. at b=0).

    Rician magnitude = sqrt((S + n_real)^2 + n_imag^2), n ~ N(0, sigma),
    sigma = S0 / SNR. Works on arrays of any shape.
    """
    rng = np.random.default_rng() if rng is None else rng
    sigma = S0 / snr
    signal = np.asarray(signal, dtype=float)
    n_real = rng.normal(0.0, sigma, size=signal.shape)
    n_imag = rng.normal(0.0, sigma, size=signal.shape)
    return np.sqrt((signal + n_real) ** 2 + n_imag ** 2)


# ---------------------------------------------------------------------------
# Core benchmark generator: fixed truth, many noise realizations
# ---------------------------------------------------------------------------
def simulate_repeats(D, Dstar, f, bvals, snr, n_noise=1000, S0=1.0, rng=None):
    """Monte Carlo over NOISE at a fixed ground truth.

    This is the workhorse for bias/precision: hold (D, D*, f) fixed, draw
    n_noise independent noisy realizations, fit each, then compare the
    distribution of estimates to the known truth.

    Returns
    -------
    dict with:
        signals : (n_noise, n_bvalues) noisy signals
        clean   : (n_bvalues,) the noise-free signal
        bvalues : (n_bvalues,)
        truth   : {"D","Dstar","f"}
        snr     : float
    """
    rng = np.random.default_rng() if rng is None else rng
    bvals = np.asarray(bvals, dtype=float)
    clean = ivim_signal(bvals, D, Dstar, f, S0=S0)
    tiled = np.broadcast_to(clean, (n_noise, bvals.size))
    signals = add_rician_noise(tiled, snr, S0=S0, rng=rng)
    return {
        "signals": signals,
        "clean": clean,
        "bvalues": bvals,
        "truth": {"D": D, "Dstar": Dstar, "f": f},
        "snr": float(snr),
    }


def simulate_random(n_voxels, bvals, snr, priors=PANCREAS_PRIORS, S0=1.0, rng=None):
    """Sample DIFFERENT random truths per voxel (for distribution-level studies).

    Returns signals (n_voxels, n_bvalues) plus the per-voxel truth arrays.
    """
    rng = np.random.default_rng() if rng is None else rng
    bvals = np.asarray(bvals, dtype=float)
    D = rng.uniform(*priors["D"], size=n_voxels)
    Dstar = rng.uniform(*priors["Dstar"], size=n_voxels)
    f = rng.uniform(*priors["f"], size=n_voxels)
    # vectorized forward model: (n_voxels, n_bvalues)
    clean = S0 * (f[:, None] * np.exp(-bvals[None, :] * Dstar[:, None])
                  + (1 - f)[:, None] * np.exp(-bvals[None, :] * D[:, None]))
    signals = add_rician_noise(clean, snr, S0=S0, rng=rng)
    return {"signals": signals, "bvalues": bvals,
            "truth": {"D": D, "Dstar": Dstar, "f": f}, "snr": float(snr)}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)  # deterministic seed -> reproducible

    bvals = B_SCHEMES["clinical_sparse"]
    t = ANCHOR_TRUTHS[0]

    clean = ivim_signal(bvals, **t)
    print("b-values :", bvals)
    print("S(b=0)   :", round(float(clean[0]), 6), "(must be 1.0)")
    print("clean    :", np.round(clean, 4))

    sim = simulate_repeats(**t, bvals=bvals, snr=20, n_noise=5000, rng=rng)
    emp = sim["signals"].mean(axis=0)
    print("\nEmpirical mean of 5000 Rician draws @ SNR=20:")
    print("  ", np.round(emp, 4))
    print("Note: Rician mean sits ABOVE the clean signal at high b — that")
    print("positive bias (the noise floor) is exactly what the fitters fight.")

    assert abs(clean[0] - 1.0) < 1e-12, "S0 normalization broken"
    assert sim["signals"].shape == (5000, bvals.size), "shape mismatch"
    print("\nSmoke test OK.")
