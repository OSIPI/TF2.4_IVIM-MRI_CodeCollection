import numpy as np
import numpy.testing as npt
import pytest
import json
import pathlib
import os
import csv
import random
import datetime

from src.wrappers.OsipiBase import OsipiBase
from utilities.data_simulation.GenerateData import GenerateData

#run using pytest <path_to_this_file> --saveFileName test_output.txt --SNR 50 100 200
#e.g. pytest -m slow tests/IVIMmodels/unit_tests/test_ivim_synthetic.py  --saveFileName test_output.csv --SNR 10 50 100 200 --fitCount 20
@pytest.mark.slow
def test_generated(ivim_algorithm, ivim_data, SNR, rtol, atol, fit_count, rician_noise, save_file, save_duration_file, use_prior):
    # assert save_file == "test"
    rng = np.random.RandomState(42)
    # random.seed(42)
    S0 = 1
    gd = GenerateData(rng=rng)
    name, bvals, data = ivim_data
    D = data["D"]
    f = data["f"]
    Dp = data["Dp"]
    fit = OsipiBase(algorithm=ivim_algorithm)
    # here is a prior
    if use_prior and hasattr(fit, "accepts_priors") and fit.accepts_priors:
        prior = [rng.normal(D, D/3, 10), rng.normal(f, f/3, 10), rng.normal(Dp, Dp/3, 10), rng.normal(1, 1/3, 10)]
        # prior = [np.repeat(D, 10)+np.random.normal(0,D/3,np.shape(np.repeat(D, 10))), np.repeat(f, 10)+np.random.normal(0,f/3,np.shape(np.repeat(D, 10))), np.repeat(Dp, 10)+np.random.normal(0,Dp/3,np.shape(np.repeat(D, 10))),np.repeat(np.ones_like(Dp), 10)+np.random.normal(0,1/3,np.shape(np.repeat(D, 10)))]  # D, f, D*
        fit.initialize(prior_in=prior)
    time_delta = datetime.timedelta()
    for idx in range(fit_count):
        # if "data" not in data:
        signal = gd.ivim_signal(D, Dp, f, S0, bvals, SNR, rician_noise)
        # else:
        #     signal = data["data"]
        start_time = datetime.datetime.now()
        [f_fit, Dp_fit, D_fit] = fit.osipi_fit(signal, bvals) #, prior_in=prior
        time_delta += datetime.datetime.now() - start_time
        if save_file is not None:
            save_file.writerow([ivim_algorithm, name, SNR, idx, f, Dp, D, f_fit, Dp_fit, D_fit, *signal])
            # save_results(save_file, ivim_algorithm, name, SNR, idx, [f, Dp, D], [f_fit, Dp_fit, D_fit])
        npt.assert_allclose([f, Dp, D], [f_fit, Dp_fit, D_fit], rtol, atol)
    if save_duration_file is not None:
        save_duration_file.writerow([ivim_algorithm, name, SNR, time_delta/datetime.timedelta(microseconds=1), fit_count])
        # save_duration(save_duration_file, ivim_algorithm, name, SNR, time_delta, fit_count)


def save_results(filename, algorithm, name, SNR, idx, truth, fit):
    with open(filename, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        data = [algorithm, name, SNR, idx, *truth, *fit]
        writer.writerow(data)

def save_duration(filename, algorithm, name, SNR, duration, count):
    with open(filename, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        data = [algorithm, name, SNR, duration/datetime.timedelta(microseconds=1), count]
        writer.writerow(data)