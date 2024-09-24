import pytest
import pathlib
import json
import csv
import tempfile
import os
import random
import numpy as np
# import datetime


def pytest_addoption(parser):
    parser.addoption(
        "--SNR",
        default=[100],
        nargs="+",
        type=int,
        help="Evaluation test SNR",
    )
    parser.addoption(
        "--ricianNoise",
        default=False,
        type=bool,
        help="Use Rician noise, non-rician is gaussian",
    )
    parser.addoption(
        "--usePrior",
        default=False,
        type=bool,
        help="Use a prior where accepted",
    )
    parser.addoption(
        "--algorithmFile",
        default="tests/IVIMmodels/unit_tests/algorithms.json",
        type=str,
        help="Algorithm file name",
    )
    parser.addoption(
        "--dataFile",
        default="tests/IVIMmodels/unit_tests/generic.json",
        type=str,
        help="Default data file name",
    )
    parser.addoption(
        "--saveFileName",
        default="",
        type=str,
        help="Saved results file name",
    )
    parser.addoption(
        "--rtol",
        default=1,
        type=float,
        help="Relative tolerance",
    )
    parser.addoption(
        "--atol",
        default=1,
        type=float,
        help="Absolute tolerance",
    )
    parser.addoption(
        "--fitCount",
        default=10,
        type=int,
        help="Number of fits to perform on the same parameters",
    )
    parser.addoption(
        "--saveDurationFileName",
        default="",
        type=str,
        help="Saved duration results file name",
    )
    parser.addoption(
        "--selectAlgorithm",
        default=[""],
        nargs="+",
        type=str,
        help="Drop all algorithms except for these from the list"
    )
    parser.addoption(
        "--dropAlgorithm",
        default=[""],
        nargs="+",
        type=str,
        help="Drop this algorithm from the list"
    )


@pytest.fixture(scope="session")
def save_file(request):
    filename = request.config.getoption("--saveFileName")
    if filename:
        current_folder = pathlib.Path.cwd()
        filename = current_folder / filename
        # print(filename)
        # filename.unlink(missing_ok=True)
        filename = filename.as_posix()

        data = data_list(request.config.getoption("--dataFile"))  # TODO: clean up this hacky way to get bvalues
        [_, bvalues, _] = next(data)
        bvalue_string = ["bval_" + str(bvalue) for bvalue in bvalues]
        # bvalue_string = ["b_0.0","b_1.0","b_2.0","b_5.0","b_10.0","b_20.0","b_30.0","b_50.0","b_75.0","b_100.0","b_150.0","b_250.0","b_350.0","b_400.0","b_550.0","b_700.0","b_850.0","b_1000.0"]

        with open(filename, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(("Algorithm", "Region", "SNR", "index", "f", "Dp", "D", "f_fitted", "Dp_fitted", "D_fitted", *bvalue_string))
            yield writer
            # writer.writerow(["", datetime.datetime.now()])
    else:
        yield None
    # return filename

@pytest.fixture(scope="session")
def save_duration_file(request):
    filename = request.config.getoption("--saveDurationFileName")
    if filename:
        current_folder = pathlib.Path.cwd()
        filename = current_folder / filename
        # print(filename)
        # filename.unlink(missing_ok=True)
        filename = filename.as_posix()
        with open(filename, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(("Algorithm", "Region", "SNR", "Duration [us]", "Count"))
            yield writer
            # writer.writerow(["", datetime.datetime.now()])
    else:
        yield None
    # return filename

@pytest.fixture(scope="session")
def rtol(request):
    return request.config.getoption("--rtol")

@pytest.fixture(scope="session")
def atol(request):
    return request.config.getoption("--atol")

@pytest.fixture(scope="session")
def fit_count(request):
    return request.config.getoption("--fitCount")

@pytest.fixture(scope="session")
def rician_noise(request):
    return request.config.getoption("--ricianNoise")

@pytest.fixture(scope="session")
def use_prior(request):
    return request.config.getoption("--usePrior")


def pytest_generate_tests(metafunc):
    if "SNR" in metafunc.fixturenames:
        metafunc.parametrize("SNR", metafunc.config.getoption("SNR"))
    if "ivim_algorithm" in metafunc.fixturenames:
        algorithms = algorithm_list(metafunc.config.getoption("algorithmFile"), metafunc.config.getoption("selectAlgorithm"), metafunc.config.getoption("dropAlgorithm"))
        metafunc.parametrize("ivim_algorithm", algorithms)
    if "ivim_data" in metafunc.fixturenames:
        data = data_list(metafunc.config.getoption("dataFile"))
        metafunc.parametrize("ivim_data", data)


def algorithm_list(filename, selected, dropped):
    current_folder = pathlib.Path.cwd()
    algorithm_path = current_folder / filename
    with algorithm_path.open() as f:
        algorithm_information = json.load(f)
    algorithms = set(algorithm_information["algorithms"])
    algorithms = algorithms - set(dropped)
    if len(selected) > 0 and selected[0]:
        algorithms = algorithms & set(selected)
    return list(algorithms)

def data_list(filename):
    current_folder = pathlib.Path.cwd()
    data_path = current_folder / filename
    with data_path.open() as f:
        all_data = json.load(f)

    bvals = all_data.pop('config')
    bvals = bvals['bvalues']
    for name, data in all_data.items():
        yield name, bvals, data

@pytest.fixture
def bval_bvec_info():
    shells = [0, 10, 20, 50, 100, 200, 500, 1000]
    # random.shuffle(shells)
    bvals = np.concatenate((shells, random.choices(shells, k=10)), axis=0)

    vecs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.707, 0.707, 0], [0.5, 0.5, 0.5], [0, 0.707, 0.707], [0.707, 0, 0.707]]
    for idx in range(len(vecs)):
        if np.linalg.norm(vecs[idx]) != 0:
            vecs[idx] = vecs[idx]/np.linalg.norm(vecs[idx])
    bvecs = []
    vecs_idx = 1  # the first index is needed for the true output, but not needed here
    for idx in range(len(bvals)):
        if bvals[idx] == 0:
            bvecs.append(np.asarray([0, 0, 0]))
        elif vecs_idx < len(vecs):
            bvecs.append(vecs[vecs_idx])
            vecs_idx += 1
        else:
            bvecs.append(random.choice(vecs[1:]))  # don't put a b0 in where it shouldn't be
    print(f'raw bvals {bvals}')
    print(f'raw bvecs {bvecs}')
    

    with tempfile.NamedTemporaryFile(mode='wt', delete=False) as fp_val, tempfile.NamedTemporaryFile(mode='wt', delete=False) as fp_vec:
        writer = csv.writer(fp_val, delimiter=' ')
        for bval in bvals:
            writer.writerow((bval,))
        fp_val.close()
        writer = csv.writer(fp_vec, delimiter=' ')
        for bvec in bvecs:
            writer.writerow(bvec)
        fp_vec.close()
        yield (fp_val.name, np.asarray(shells), bvals, fp_vec.name, np.asarray(vecs), np.asarray(bvecs))
        os.unlink(fp_val.name)  # use NamedTemporaryFile with delete_on_close with later python versions
        os.path.exists(fp_val.name)
        os.unlink(fp_vec.name)  # use NamedTemporaryFile with delete_on_close with later python versions
        os.path.exists(fp_vec.name)
        


