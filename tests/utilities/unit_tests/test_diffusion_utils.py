import numpy as np
import numpy.testing as npt
from utilities.process.file_io import read_bval_file, read_bvec_file
from utilities.process.diffusion_utils import find_shells, find_directions, normalize_series
    

#TODO: test without b0
#TODO: test symmetry
def test_read_bval_bvec(bval_bvec_info):
    bval_name, shells, bvals, bvec_name, directions, bvecs = bval_bvec_info
    saved_bvals = read_bval_file(bval_name)
    npt.assert_equal(bvals, np.asarray(saved_bvals))
    saved_shells, bval_indices, b0 = find_shells(saved_bvals)
    npt.assert_equal(shells, saved_shells, "Shells do not match")
    npt.assert_equal(saved_bvals, [saved_shells[index] for index in bval_indices], "Bvalue indices are incorrect")

    saved_bvecs = read_bvec_file(bvec_name)
    npt.assert_allclose(np.asarray(bvecs), np.asarray(saved_bvecs), err_msg="Incorrectly saved bvectors")
    vectors, bvec_indices, groups = find_directions(saved_bvecs, b0)
    assert vectors.shape[0] == groups.shape[1] + 1, "Number of vectors is correct"
    assert vectors.shape == np.asarray(directions).shape, "Number of elements in directions does not match"
    directions_set = set()
    for direction in directions:
        directions_set.add(tuple(direction))
    vectors_set = set()
    for vector in vectors:
        vectors_set.add(tuple(vector))
    assert directions_set == vectors_set, "Elements in directions does not match"
    npt.assert_equal(saved_bvecs, [vectors[index] for index in bvec_indices], "Bvector indices are incorrect")

def test_normalization():
    original = np.atleast_2d([[10, 10], [10, 10], [5, 5], [5, 5]]).T

    indices = [True, False, False, False]
    updated = normalize_series(original.copy(), indices)
    npt.assert_allclose(original / 10, updated, err_msg="Normalization with 1 point failed")

    indices = [True, True, False, False]
    updated = normalize_series(original.copy(), indices)
    npt.assert_allclose(original / 10, updated, err_msg="Normalization with 2 points failed")

    indices = [False, True, True, False]
    updated = normalize_series(original.copy(), indices)
    npt.assert_allclose(original / 7.5, updated, err_msg="Normalization with 2 different points failed")

    indices = [False, False, False, True]
    updated = normalize_series(original.copy(), indices)
    npt.assert_allclose(original / 5, updated, err_msg="Normalization with 1 final point failed")

    original = np.asarray([10, 5])
    indices = [True, False]

    updated = normalize_series(original.copy(), indices)
    npt.assert_allclose(original / 10, updated, err_msg="Normalization of 1D failed")
