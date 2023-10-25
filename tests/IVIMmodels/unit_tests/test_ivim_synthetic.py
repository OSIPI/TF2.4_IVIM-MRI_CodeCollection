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
def test_generated(ivim_algorithm, ivim_data, SNR, rtol, atol, fit_count, save_file, save_duration_file):
    # assert save_file == "test"
    random.seed(42)
    S0 = 1
    gd = GenerateData()
    name, bvals, data = ivim_data
    D = data["D"]
    f = data["f"]
    Dp = data["Dp"]
    fit = OsipiBase(algorithm=ivim_algorithm)
    time_delta = datetime.timedelta()
    for idx in range(fit_count):
        # if "data" not in data:
        signal = gd.ivim_signal(D, Dp, f, S0, bvals, SNR)
        # else:
        #     signal = data["data"]
        start_time = datetime.datetime.now()
        [f_fit, Dp_fit, D_fit] = fit.osipi_fit(signal, bvals)
        time_delta += datetime.datetime.now() - start_time
        if save_file:
            save_results(save_file, ivim_algorithm, name, SNR, [f, Dp, D], [f_fit, Dp_fit, D_fit])
        npt.assert_allclose([f, Dp, D], [f_fit, Dp_fit, D_fit], rtol, atol)
    if save_duration_file:
        save_duration(save_duration_file, ivim_algorithm, name, SNR, time_delta, fit_count)


def save_results(filename, algorithm, name, SNR, truth, fit):
    with open(filename, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        data = [algorithm, name, SNR, *truth, *fit]
        writer.writerow(data)

def save_duration(filename, algorithm, name, SNR, duration, count):
    with open(filename, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        data = [algorithm, name, SNR, duration/datetime.timedelta(microseconds=1), count]
        writer.writerow(data)