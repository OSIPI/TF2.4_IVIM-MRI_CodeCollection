import numpy as np


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    nv1 = np.linalg.norm(v1)
    nv2 = np.linalg.norm(v2)
    if nv1 == 0 or nv2 == 0:
        return 0
    v1_u = v1 / nv1
    v2_u = v2 / nv2
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def find_shells(bvals_raw, rtol=0.01, atol=1):
    """ Automatically finds shells from a bvalue file.
    
    Given a raw list of bvalues and optional relative and absolute tolerances,
    return the unique shells, shell indices, and b0 image indices.

    Parameters
    ------------
    bvals_raw : list
        List of bvalues as acquired
    rtol : float
        Relative tolerance of bvalues
    atol : float
        Absolute tolerance of bvalues

    Returns
    ----------
    shells : list
        The mean values of the shells
    indices : list
        The indices of the shells in the original bvalue list
    b0 : list
        A boolean array of locations of b0 values
    """
    # if we know the number of values in each shell we could use np.partition
    # but I don't think that's very useful.

    bvals_raw = np.asarray(bvals_raw)
    assert len(bvals_raw.shape) == 1, "Must be a 1D array"

    # group the bvalues based on tolerances
    bvals_indices = np.argsort(bvals_raw)  # sort order
    bvals = bvals_raw[bvals_indices]
    index = np.zeros_like(bvals, dtype=int)
    idx_lower = 0
    for idx_higher in range(1, len(bvals)):
        if np.allclose(bvals[idx_lower], bvals[idx_higher], rtol, atol):
            index[idx_higher] = index[idx_lower]
        else:
            index[idx_higher] = index[idx_lower] + 1
            idx_lower = idx_higher

    # invert the sort to return the shell indices of the unsorted bvalues
    bvals_inverse_sort = np.empty_like(bvals_indices)
    bvals_inverse_sort[bvals_indices] = np.arange(bvals_indices.size)

    # find the mean b-values of the shells
    shells = [np.mean(bvals[index==idx]) for idx in set(index)]
    indices = index[bvals_inverse_sort]
    return shells, indices, indices<atol #, bvals_indices

def find_directions(bvecs_raw, zero_indices, atol=1, rotationally_symmetric=True):
    """ Automatically find the direction groups

    Given a list of bvecs, b0 indices, and optionally absolute tolerance, and rotationally symmetric flag,
    group the directions from the bvec file into distinct direction sets.
    Similar to the bvalue shell grouping but for the directions.

    Parameters
    ----------
    bvecs_raw : 2D numpy array
        Numpy array where the rows are the bvecs of the acquisition
    zero_indices : list
        List indicating which rows of bvecs_raw are b0 so the vector can be ignored for this step
    atol : float
        Absolute angle tolerance in degrees
    rotationally_symmetric : boolean
        Should the grouping treat opposite direction angles as the same?

    Returns
    -------
    vectors : 2D numpy array
        The mean vector for each direction group. If present, the zero vector is first, group 0.
    indices : list
        The indices of the direction group in the original bvalue list
    """

    bvecs_raw = np.asarray(bvecs_raw)
    zero_indices = np.asarray(zero_indices)
    assert len(bvecs_raw.shape) == 2, "Must be a 2D array"
    assert bvecs_raw.shape[1] == 3, "Must be Nx3"
    assert bvecs_raw.shape[0] == len(zero_indices), "The length of vectors and B0 indices must match"

    # drop b=0 images and generate a unique set
    bvecs_unique = set()
    bvecs = bvecs_raw[~zero_indices]
    for bvec in bvecs:
        bvecs_unique.add(tuple(bvec))

    # iterate the unique bvectors and add them to the matching dictionary group
    # the key is the vector the value is the group number
    bvecs_groups = {}
    if np.any(zero_indices):
        next_idx = 1
    else:
        next_idx = 0
    bvecs_groups[bvecs_unique.pop()] = next_idx
    next_idx += 1
    for bvec in bvecs_unique:
        for group_bvec, group_id in bvecs_groups.items():
            angle = angle_between(bvec, group_bvec)
            if angle < atol or (abs(angle - 180) < atol and rotationally_symmetric):
                bvecs_groups[bvec] = group_id
                break
        else:
            bvecs_groups[bvec] = next_idx
            next_idx += 1

    # match the groups back to indices of the full bvec list
    indices_nonzero = np.zeros(bvecs.shape[0], dtype=int)
    for idx in range(bvecs.shape[0]):
        indices_nonzero[idx] = bvecs_groups[tuple(bvecs[idx])]
    indices = np.zeros(bvecs_raw.shape[0], dtype=int)
    indices[~zero_indices] = indices_nonzero
    
    # print(f'bvec indices {indices}')
    
    # reverse the map to go from index to group id
    # the key is the group number and the value is the bvalue
    groups_bvecs = {}
    for k, v in bvecs_groups.items():
        groups_bvecs[v] = groups_bvecs.get(v, []) + [k]
    num_vectors = len(groups_bvecs)
    # print(f'groups_bvecs {groups_bvecs}')

    # find the mean value of each group
    if np.any(zero_indices):
        vectors = np.zeros((len(groups_bvecs) + 1, 3))
        # vectors[0] = np.asarray([0, 0, 0])
    else:
        vectors = np.zeros((len(groups_bvecs), 3))
    for idx, group in groups_bvecs.items():
        group = np.asarray(group)
        vectors[idx] = np.mean(group, 0)

    
    groups = np.zeros([len(indices), num_vectors], dtype=bool)
    # print(f"groups.shape {groups.shape}")
    for num in range(num_vectors):
        # print(f"num {num + 1} where {(indices == 0) | (indices == (num + 1))}")
        groups[:, num] = (indices == 0) | (indices == (num + 1))
    
        
    # vector_length_with_b0 = np.count_nonzero(zero_indices) + len(groups_bvecs[1])
    return vectors, indices, groups #, direction_groups


def normalize_series(img, indices, axis=None):
    """ Normalize a series of images based on one or more indices

    Given an image and desired indices, normalize the images in the input.

    Parameters
    ----------
    img : 2+D numpy array
        Numpy array to normalize
    indices : 1D iterable
        True indices indicate the values which are the normalizers
    axis : int
        The axis to normalize on. None automatically chooses the last.

    Returns
    -------
    vectors : 2+D numpy array
        The same array just normalized to the input values
    """
    assert np.any(indices), "There are no true values in the indices"

    if axis is None:
        axis = img.ndim - 1
    num_axis = img.shape[axis]
    
    assert num_axis == len(indices), "The number of images and indices do not match"

    b0 = np.mean(img[..., indices], axis=axis)
    newshape = list(b0.shape)
    newshape.extend([1]*(img.ndim - len(newshape)))
    b0 = np.reshape(b0, newshape)
    normalizer = np.repeat(b0, num_axis, axis=axis)
    return img / normalizer

