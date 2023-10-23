import numpy as np
import numpy.testing as npt
import pytest
import json
import pathlib
import os
import csv
import random

from src.wrappers.OsipiBase import OsipiBase
from utilities.data_simulation.GenerateData import GenerateData

#run using pytest <path_to_this_file> --saveFileName test_output.txt --SNR 50 100 200
#e.g. pytest tests/IVIMmodels/unit_tests/test_ivim_synthetic.py --saveFileName test_output.txt --SNR 50 100 200
def test_generated(ivim_algorithm, ivim_data, SNR, rtol, atol, fit_count, save_file):
    # assert save_file == "test"
    random.seed(42)
    S0 = 1
    gd = GenerateData()
    name, bvals, data = ivim_data
    D = data["D"]
    f = data["f"]
    Dp = data["Dp"]
    fit = OsipiBase(algorithm=ivim_algorithm)
    for idx in range(fit_count):
        # if "data" not in data:
        signal = gd.ivim_signal(D, Dp, f, S0, bvals, SNR)
        # else:
        #     signal = data["data"]
        [f_fit, Dp_fit, D_fit] = fit.ivim_fit(signal, bvals)
        if save_file:
            save_results(save_file, ivim_algorithm, name, SNR, [f, D, Dp], [f_fit, Dp_fit, D_fit])
        npt.assert_allclose([f, D, Dp], [f_fit, D_fit, Dp_fit], rtol, atol)


def save_results(filename, algorithm, name, SNR, truth, fit):
    with open(filename, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        data = [algorithm, name, SNR, truth, fit]
        writer.writerow(data)