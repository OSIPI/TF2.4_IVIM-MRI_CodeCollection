import pytest
import pathlib
import json
import csv
import datetime


def pytest_addoption(parser):
    parser.addoption(
        "--SNR",
        default=[100],
        nargs="+",
        type=int,
        help="Evaluation test SNR",
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


@pytest.fixture(scope="session")
def save_file(request):
    filename = request.config.getoption("--saveFileName")
    if filename:
        current_folder = pathlib.Path.cwd()
        filename = current_folder / filename
        # print(filename)
        # filename.unlink(missing_ok=True)
        filename = filename.as_posix()
        with open(filename, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter='#')
            writer.writerow(["", datetime.datetime.now()])
    return filename

@pytest.fixture(scope="session")
def rtol(request):
    return request.config.getoption("--rtol")


@pytest.fixture(scope="session")
def atol(request):
    return request.config.getoption("--atol")


def pytest_generate_tests(metafunc):
    if "SNR" in metafunc.fixturenames:
        metafunc.parametrize("SNR", metafunc.config.getoption("SNR"))
    if "ivim_algorithm" in metafunc.fixturenames:
        algorithms = algorithm_list(metafunc.config.getoption("algorithmFile"))
        metafunc.parametrize("ivim_algorithm", algorithms)
    if "algorithm_file" in metafunc.fixturenames:
        metafunc.parametrize("algorithm_file", metafunc.config.getoption("algorithmFile"))
    if "ivim_data" in metafunc.fixturenames:
        data = data_list(metafunc.config.getoption("dataFile"))
        metafunc.parametrize("ivim_data", data)
    if "data_file" in metafunc.fixturenames:
        metafunc.parametrize("data_file", metafunc.config.getoption("dataFile"))


def algorithm_list(filename):
    current_folder = pathlib.Path.cwd()
    algorithm_path = current_folder / filename
    with algorithm_path.open() as f:
        algorithm_information = json.load(f)
    return algorithm_information["algorithms"]

def data_list(filename):
    current_folder = pathlib.Path.cwd()
    data_path = current_folder / filename
    with data_path.open() as f:
        all_data = json.load(f)

    bvals = all_data.pop('config')
    bvals = bvals['bvalues']
    for name, data in all_data.items():
        yield name, bvals, data