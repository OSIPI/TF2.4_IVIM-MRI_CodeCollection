import argparse
from utilities.process.file_io import read_nifti_file, read_bval_file, read_bvec_file, save_nifti_file
from utilities.process.diffusion_utils import find_directions, find_shells, normalize_series
from src.wrappers.OsipiBase import OsipiBase
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial



def loop_over_first_n_minus_1_dimensions(arr):
    """
    Loops over the first n-1 dimensions of a numpy array.

    Args:
        arr: A numpy array.

    Yields:
        A tuple containing the indices for the current iteration and a flattened view of the remaining dimensions.
    """
    n = arr.ndim
    for idx in np.ndindex(*arr.shape[:n-1]):
        flat_view = arr[idx].flatten()
        yield idx, flat_view

def generate_data(data, bvals, b0_indices, groups, total_iteration):
    """
    Generates data samples for a multiprocess fitting

    Args:
        data: The raw data to be sampled
        bvals: The bvalues
        b0_indices: The b0 indices in the data
        groups: The group indices in the data
        total_iterations: The total number of iterations to generate

    Yields:
        A tuple containing matching: data, bvalues, and b0_indices
    """
    num_directions = groups.shape[1]
    data = data.reshape(total_iteration, -1)
    for idx in range(total_iteration):
        for dir in range(num_directions):
            # print('yielding')
            yield (data[idx, groups[:, dir]].flatten(), bvals[:, groups[:, dir]].ravel(), b0_indices[:, groups[:, dir]].ravel())

def osipi_fit(fitfunc, data_bvals):
    """
    Fit the data using the provided fit function

    Args:
        fitfunc: The fit function
        data_bvals: The tuple of data, bvals, and b0_indices

    Returns:
        The fitted values
    """
    data, bvals, b0_indices = data_bvals
    data = normalize_series(data, b0_indices)
    # print(f'data.shape {data.shape} data {data} bvals {bvals}')
    return fitfunc(data, bvals)

# def osipi_fit(fitfunc, bvals, data, f_image, Dp_image, D_image, index):
#     bval_index = len(f_image) % len(bvals)
#     print(f'data.shape {data.shape} index {index} data[index] {data[index]} bvals.shape {bvals.shape} bval_index {bval_index} bvals {bvals[:, bval_index]}')
#     [f_fit, Dp_fit, D_fit] = fitfunc(data[index], bvals[:, bval_index])
#     f_image[index] = f_fit
#     Dp_image[index] = Dp_fit
#     D_image[index] = D_fit



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a 4D NIfTI phantom file along with BIDS JSON, b-vector, and b-value files.")
    parser.add_argument("input_file", type=str, help="Path to the input 4D NIfTI file.")
    parser.add_argument("bvec_file", type=str, help="Path to the b-vector file.")
    parser.add_argument("bval_file", type=str, help="Path to the b-value file.")
    parser.add_argument("--nproc", type=int, default=0, help="Number of processes to use, -1 disabled multprocessing, 0 automatically determines number, >0 uses that number.")
    parser.add_argument("--group_directions", default=False, action="store_true", help="Fit all directions together")
    parser.add_argument("--affine", type=float, default=None, nargs="+", help="Affine matrix for NIfTI image.")
    parser.add_argument("--algorithm", type=str, default="OJ_GU_seg", help="Select the algorithm to use.")
    parser.add_argument("--algorithm_args", default={}, nargs=argparse.REMAINDER, help="Additional arguments for the algorithm.")
    

    args = parser.parse_args()

    try:
        # Read the 4D NIfTI file
        data, _ = read_nifti_file(args.input_file)
        data = data[0::4, 0::4, 0::2, :]
        print(f'data.shape {data.shape}')

        # Read the b-vector, and b-value files
        bvecs = read_bvec_file(args.bvec_file)
        bvals = read_bval_file(args.bval_file)
        # print(f'bvals.size {bvals.shape} bvecs.size {bvecs.shape}')
        print(bvals)
        print(bvecs)
        shells, bval_indices, b0_indices = find_shells(bvals)
        num_b0 = np.count_nonzero(b0_indices)
        print(shells)
        print(bval_indices)
        # print(b0_indices)
        # print(bvecs)

        # print('vectors')
        vectors, bvec_indices, groups = find_directions(bvecs, b0_indices)
        print(vectors)
        print(bvec_indices)
        print(f'groups {groups}')

        # split_bval_bvec(bvec_indices, num_vectors)
        # quit()
        

        # Pass additional arguments to the algorithm
        fit = OsipiBase(algorithm=args.algorithm, **args.algorithm_args)

        # n = data.ndim
        output_shape = list(data.shape[:-1])
        # if args.group_directions:
        #     input_data = data
        #     input_bvals = np.atleast_2d(bvals)
        # else:
        #     num_directions = groups.shape[1]
        #     measurements = np.count_nonzero(groups[:, 0])
        #     print(f"group_length {num_directions}")
        #     input_shape = output_shape.copy()

        #     print(f"groups[:, 0] {groups[:,0]} {np.count_nonzero(groups[:, 0])}")
        #     input_shape.append(num_directions)
        #     input_shape.append(measurements)
        #     output_shape.append(num_directions)
        #     print(f"input_shape {input_shape}")
        #     input_data = np.zeros(input_shape)
        #     input_bvals = np.zeros([measurements, num_directions])
        #     for group_idx in range(num_directions):
        #         print(f"group {group_idx} {groups[:, group_idx]}")
        #         input_data[..., group_idx, :] = data[..., groups[:, group_idx]]
        #         input_bvals[:, group_idx] = bvals[groups[:, group_idx]]
        # if args.group_directions:
        #     input_data = data
        #     input_bvals = np.atleast_2d(bvals)
        # else:
        input_data = data
        input_bvals = np.atleast_2d(bvals)
        b0_indices = np.atleast_2d(b0_indices)
        print(f"data.shape {data.shape}")
        print(f"input_data.shape {input_data.shape}")
            
            
                
        
        voxel_iteration = np.prod(output_shape)
        group_iteration = groups.shape[1]
        total_iteration = voxel_iteration * group_iteration
        output_shape.append(group_iteration)
        f_image = np.zeros(output_shape)
        Dp_image = np.zeros(output_shape)
        D_image = np.zeros(output_shape)
        print(f_image.shape)

        # This is necessary for the tqdm to display progress bar.
        
        # total_iteration = np.prod(data.shape[:n-1])
        print(f'input_bvals {input_bvals}')
        print(f'voxel_iteration {voxel_iteration} input_data.shape {input_data.shape}')
        # print(f'input_data[5000] {input_data.reshape(total_iteration, -1)[5000]}')

        
        # fit_partial = partial(osipi_fit, fit.osipi_fit, input_bvals, input_data.reshape(total_iteration, -1), f_image.reshape(total_iteration), Dp_image.reshape(total_iteration), D_image.reshape(total_iteration))
        fit_partial = partial(osipi_fit, fit.osipi_fit)

        
        if args.nproc >= 0:
            print('multiprocess fitting')
            gd = generate_data(input_data, input_bvals, b0_indices, groups, voxel_iteration)
            map_args = [fit_partial, gd]
            chunksize = round(total_iteration / args.nproc) if args.nproc > 0 else round(total_iteration / 128)
            print(f'chunksize {chunksize}')
            map_kwargs = {'desc':f"{args.algorithm} is fitting", 'dynamic_ncols':True, 'total':total_iteration, 'chunksize':chunksize}
            if args.nproc > 0:
                map_kwargs['max_workers'] = args.nproc
            result = process_map(*map_args, **map_kwargs)
            output = np.asarray(result)
            print(f'output.shape {output.shape}')
            output = output.reshape([*output_shape, 3])
            f_image = output[..., 0]
            print(f'f_img.shape {f_image.shape}')
            Dp_image = output[..., 1]
            D_image = output[..., 2]
            # print(result)
            # if args.nproc == 0:  # TODO: can this be done more elegantly, I just want to omit a single parameter here
            #     process_map(fit_partial, range(total_iteration), desc=f"{args.algorithm} is fitting", dynamic_ncols=True, total=total_iteration)
            # else:
            #     process_map(fit_partial, range(total_iteration), max_workers=args.nproc, desc=f"{args.algorithm} is fitting", dynamic_ncols=True, total=total_iteration)
        else:
            for idx, view in tqdm(loop_over_first_n_minus_1_dimensions(data), desc=f"{args.algorithm} is fitting", dynamic_ncols=True, total=total_iteration):
                [f_fit, Dp_fit, D_fit] = fit.osipi_fit(view, bvals)
                f_image[idx] = f_fit
                Dp_image[idx] = Dp_fit
                D_image[idx] = D_fit

        print("finished fitting")

        save_nifti_file(f_image, "f.nii.gz", args.affine)
        save_nifti_file(Dp_image, "dp.nii.gz", args.affine)
        save_nifti_file(D_image, "d.nii.gz", args.affine)

    except Exception as e:
        print(f"Error: {e}")

