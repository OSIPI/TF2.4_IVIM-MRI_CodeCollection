import numpy as np
import pytest
import time
from src.wrappers.OsipiBase import OsipiBase
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
#run using python -m pytest from the root folder


def test_parallel(algorithmlist,eng,threeddata, results_file="parallel_timing_results.xlsx"):
    algorithm, requires_matlab = algorithmlist
    data, Dim, fim, Dpim, bvals = threeddata
    # Get index of b=0
    b0_index = np.where(bvals == 0)[0][0]
    data[data < 0] = 0
    # Mask of voxels where signal at b=0 >= 0.5
    invalid_mask = data[:, :, :, b0_index] < 0.01
    data[invalid_mask,:] = np.nan
    print('testing ' + str(np.sum(~invalid_mask)) + ' voxels of a matrix size ' + str(np.shape(data)))
    if requires_matlab:
        if eng is None:
            pytest.skip(reason="Running without matlab; if Matlab is available please run pytest --withmatlab")
        else:
            kwargs = {'eng': eng}
    else:
        kwargs={}
    fit = OsipiBase(algorithm=algorithm,**kwargs)

    def dummy_task(x):
        return x

    start_time = time.time()  # Record the start time
    fit_result = fit.osipi_fit(data, bvals,njobs=1)
    serial_time = time.time() - start_time  # Calculate elapsed time
    row = {
        "algorithm": algorithm,
        "voxels_tested": int(np.sum(~invalid_mask)),
        "serial_time_s": round(serial_time, 4),
    }
    for n_jobs in [2, 4, 8]:
        Parallel(n_jobs=n_jobs)(delayed(dummy_task)(i) for i in range(32))  # github actions only supports 2 cores
        start_time = time.time()  # Record the start time
        fit_result2 = fit.osipi_fit(data, bvals,njobs=n_jobs)
        parallel_time = time.time() - start_time  # Calculate elapsed time
        # --- Save timing results to Excel ---
        row[f"parallel_{n_jobs}_s"] = round(parallel_time, 4)
        row[f"speedup_{n_jobs}x"] = round(serial_time / parallel_time if parallel_time > 0 else np.nan, 3)

    results_path = Path(results_file).with_suffix(".csv")
    if results_path.exists():
        existing = pd.read_csv(results_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(results_path, index=False)

    # ============== Plotting All Results ==============
    df = pd.read_csv(results_path)

    core_counts = [1, 2, 4, 8]

    # --- Plot computing time vs cores for all algorithms ---
    plt.figure(figsize=(7, 5))
    for algo, group in df.groupby("algorithm"):
        times = [group["serial_time_s"].mean()] + [
            group[f"parallel_{c}_s"].mean() for c in [2, 4, 8]
        ]
        plt.plot(core_counts, times, marker="o", label=algo)

    plt.xticks(core_counts)
    plt.xlabel("Number of cores (n_jobs)")
    plt.ylabel("Computing time (s)")
    plt.title("Computing time vs. cores (all algorithms)")
    plt.grid(True)
    plt.legend()
    plt.savefig("time_vs_cores_all.png", dpi=150)
    plt.close()

    # --- Plot speed-up vs cores for all algorithms ---
    plt.figure(figsize=(7, 5))
    for algo, group in df.groupby("algorithm"):
        speedups = [
            group[f"speedup_{c}x"].mean() for c in [2, 4, 8]
        ]
        plt.plot([2, 4, 8], speedups, marker="o", label=algo)

    plt.xticks([2, 4, 8])
    plt.xlabel("Number of cores (n_jobs)")
    plt.ylabel("Speed-up (serial/parallel)")
    plt.title("Speed-up vs. cores (all algorithms)")
    plt.grid(True)
    plt.legend()
    plt.savefig("speedup_vs_cores_all.png", dpi=150)
    plt.close()