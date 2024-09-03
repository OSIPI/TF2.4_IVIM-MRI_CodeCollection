import tempfile
import os
import numpy as np
import numpy.testing as npt
from utilities.process.file_io import save_nifti_file, read_nifti_file, read_bval_file, read_bvec_file


def test_nifti_read_write():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'my_nifti.nii.gz')
        data = np.random.rand(7, 8, 9)
        save_nifti_file(data, path)
        assert os.path.exists(path), "Nifti file does not exist"
        saved_data, saved_hdr = read_nifti_file(path)
        npt.assert_equal(data, saved_data, "Nifti data does not match")

def test_read_bval_bvec(bval_bvec_info):
    bval_name, shells, bvals, bvec_name, directions, bvecs = bval_bvec_info
    assert bvecs.shape[1] == 3, "Bvec input is not Nx3"
    saved_bvals = read_bval_file(bval_name)
    npt.assert_equal(bvals, np.asarray(saved_bvals), "Bvalues do not match")
    saved_bvecs = read_bvec_file(bvec_name)
    npt.assert_allclose(bvecs, np.asarray(saved_bvecs), err_msg="Bvectors do not match")
