"""
Regression tests for the DL full-volume off-by-one fix in ivim_fit.fit_batch.

The real failure (8 NaN cells in the v2 grid): a low-SNR voxel normalizes to a
non-finite row, the DL wrapper drops it and returns N-1 predictions, its
np.reshape(..., (N,)) raises, OsipiBase.osipi_fit_full_volume swallows that into
a `False`, and fit_batch's old path-2 then dumped the WHOLE cell to the per-voxel
path -> all NaN.

We reproduce the wrapper's exact contract with a mock (no 6-min net train needed)
and assert the post-fix invariant: fit_batch always returns arrays of length N,
NaN-ing only the genuinely bad voxel.
"""
import os, sys
import numpy as np
import pytest

# repo root (parent of the uq package) on sys.path so `import uq.*` resolves and
# uq/__init__ adds the root for the upstream `src` reach-through.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from uq.ivim_fit import fit_batch


class _MockDLWrapper:
    """Mimics a deep-learning OSIPI wrapper + OsipiBase.osipi_fit_full_volume.

    - osipi_fit raises  -> forces fit_batch onto the full-volume path (path 2)
    - has ivim_fit_full_volume attr -> fit_batch treats it as full-volume capable
    - osipi_fit_full_volume drops non-finite rows then reshapes to the FULL count,
      exactly like IVIM_NEToptim.ivim_fit_full_volume (the 199-into-200 bug), and
      returns False on the resulting reshape error (OsipiBase's own behaviour).
    """
    def ivim_fit_full_volume(self, signals, **kw):   # presence => "supports full volume"
        return True

    def osipi_fit(self, sig):
        raise RuntimeError("DL methods have no classical batched fit")

    def osipi_fit_full_volume(self, data, **kw):
        data = np.atleast_2d(np.asarray(data, float))
        n = data.shape[0]
        try:
            finite = np.isfinite(data).all(axis=1)
            par = data[finite].mean(axis=1)          # 'prediction' for kept voxels only
            D = np.reshape(par, (n,))                 # BUG: reshape to full N -> raises if any dropped
            return {"D": D, "Dp": D * 2, "f": D * 3}
        except Exception:
            return False                              # OsipiBase swallows the wrapper error


class _MockClassical1D:
    """A 1D-only wrapper with NO full-volume support: path 1 fails on a batch,
    path 3 (per-voxel) must still work. Guards against the chunked helper
    hijacking non-DL methods into an all-NaN cell."""
    def osipi_fit(self, sig):
        sig = np.asarray(sig, float)
        if sig.ndim == 2:
            raise RuntimeError("no batched fit")      # force per-voxel path
        return {"D": 1e-3, "Dp": 1e-2, "f": 0.1}


def _clean_batch(n=20, nb=8):
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(0.5, 0.1, size=(n, nb)))


def test_clean_dl_batch_all_finite():
    """No bad rows -> every voxel returned, all finite, correct length."""
    sig = _clean_batch(20)
    D, Ds, F = fit_batch(_MockDLWrapper(), sig)
    assert len(D) == len(Ds) == len(F) == 20
    assert np.isfinite(D).all() and np.isfinite(Ds).all() and np.isfinite(F).all()


@pytest.mark.parametrize("bad_idx", [0, 7, 19])
def test_one_bad_row_isolated_not_whole_cell(bad_idx):
    """One non-finite row used to NaN the whole cell; now exactly that voxel is NaN."""
    sig = _clean_batch(20)
    sig[bad_idx, 3] = np.nan                          # the dropped voxel
    D, Ds, F = fit_batch(_MockDLWrapper(), sig)

    assert len(D) == 20, "length invariant: must return one row per input voxel"
    assert not np.isfinite(D[bad_idx]), "the bad voxel must be NaN"
    good = np.ones(20, bool); good[bad_idx] = False
    assert np.isfinite(D[good]).all(), "all OTHER voxels must survive (not an all-NaN cell)"
    assert int(np.isfinite(D).sum()) == 19


def test_multiple_bad_rows():
    sig = _clean_batch(50)
    bad = [3, 17, 18, 49]
    for b in bad:
        sig[b, 2] = np.inf
    D, Ds, F = fit_batch(_MockDLWrapper(), sig)
    assert len(D) == 50
    assert int(np.isfinite(D).sum()) == 50 - len(bad)
    assert not np.isfinite(D[bad]).any()


def test_classical_1d_fallback_untouched():
    """Non-DL, no full-volume: chunked helper must bow out (return None) so the
    per-voxel path runs and produces finite results."""
    sig = _clean_batch(10)
    D, Ds, F = fit_batch(_MockClassical1D(), sig)
    assert len(D) == 10
    assert np.isfinite(D).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
