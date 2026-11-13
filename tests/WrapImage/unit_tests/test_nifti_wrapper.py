import runpy

import nibabel as nib
import numpy as np
import pytest

from WrapImage import nifti_wrapper

BVALUES = np.array([0, 10, 20, 50, 100, 200, 400, 800], dtype=float)
AFFINE = np.array([[2, 0, 0, 5], [0, 2, 0, 6], [0, 0, 3, 7], [0, 0, 0, 1]], dtype=float)


@pytest.fixture
def dataset(tmp_path):
    """Writes a small noise-free IVIM image together with its bval and bvec files."""
    signal = 0.1 * np.exp(-BVALUES * 0.01) + 0.9 * np.exp(-BVALUES * 0.001)
    data = np.tile(signal, (3, 3, 2, 1)) * 1000
    nib.save(nib.Nifti1Image(data, AFFINE), tmp_path / "in.nii.gz")
    np.savetxt(tmp_path / "in.bval", BVALUES[None], fmt="%g")
    np.savetxt(tmp_path / "in.bvec", np.zeros((3, BVALUES.size)), fmt="%g")
    return tmp_path


def run(dataset, *extra):
    nifti_wrapper.main([
        str(dataset / "in.nii.gz"), str(dataset / "in.bvec"), str(dataset / "in.bval"),
        "--output_dir", str(dataset / "out"), *extra,
    ])


def test_outputs_keep_input_affine(dataset):
    run(dataset)
    for name in ("f", "dp", "d"):
        image = nib.load(dataset / "out" / f"{name}.nii.gz")
        assert image.shape == (3, 3, 2)
        np.testing.assert_allclose(image.affine, AFFINE)
    assert nib.load(dataset / "out" / "d.nii.gz").get_fdata().mean() == pytest.approx(0.001, abs=5e-4)


def test_affine_argument_overrides_input_affine(dataset):
    run(dataset, "--affine", *np.eye(4).flatten().astype(str))
    np.testing.assert_allclose(nib.load(dataset / "out" / "f.nii.gz").affine, np.eye(4))


def test_mask_restricts_fit(dataset):
    mask = np.zeros((3, 3, 2))
    mask[1, 1, :] = 1
    nib.save(nib.Nifti1Image(mask, AFFINE), dataset / "mask.nii.gz")
    run(dataset, "--mask", str(dataset / "mask.nii.gz"))
    d = nib.load(dataset / "out" / "d.nii.gz").get_fdata()
    assert np.all(d[mask == 0] == 0)
    assert np.all(d[mask == 1] > 0)


def test_mask_shape_mismatch_raises(dataset):
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), AFFINE), dataset / "mask.nii.gz")
    with pytest.raises(ValueError, match="Mask shape"):
        run(dataset, "--mask", str(dataset / "mask.nii.gz"))


def test_bvalue_count_mismatch_raises(dataset):
    np.savetxt(dataset / "in.bval", BVALUES[None, :-1], fmt="%g")
    with pytest.raises(ValueError, match="b-values"):
        run(dataset)


def test_cli_exits_nonzero_on_error(dataset, monkeypatch):
    monkeypatch.setattr("sys.argv", ["nifti_wrapper", str(dataset / "missing.nii.gz"), "x", "y"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("WrapImage.nifti_wrapper", run_name="__main__")
    assert excinfo.value.code == 1

