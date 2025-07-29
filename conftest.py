import pytest
import pathlib
import json
import csv
# import datetime
import numpy as np
from phantoms.MR_XCAT_qMRI.sim_ivim_sig import phantom

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
    parser.addoption(
        "--withmatlab",
        action="store_true",
        default=False,
        help="Run MATLAB-dependent tests"
    )


@pytest.fixture(scope="session")
def eng(request):
    """Start and return a MATLAB engine session if --withmatlab is set."""
    if not request.config.getoption("--withmatlab"):
        return None
    import matlab.engine
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    print("MATLAB engine started.")
    return eng


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
    if "ivim_data" in metafunc.fixturenames:
        data = data_list(metafunc.config.getoption("dataFile"))
        metafunc.parametrize("ivim_data", data)
    if "data_ivim_fit_saved" in  metafunc.fixturenames:
        args = data_ivim_fit_saved(metafunc.config.getoption("dataFile"),metafunc.config.getoption("algorithmFile"))
        metafunc.parametrize("data_ivim_fit_saved", args)
    if "algorithmlist" in metafunc.fixturenames:
        args = algorithmlist(metafunc.config.getoption("algorithmFile"))
        metafunc.parametrize("algorithmlist", args)
    if "bound_input" in metafunc.fixturenames:
        args = bound_input(metafunc.config.getoption("dataFile"),metafunc.config.getoption("algorithmFile"))
        metafunc.parametrize("bound_input", args)


def data_list(filename):
    current_folder = pathlib.Path.cwd()
    data_path = current_folder / filename
    with data_path.open() as f:
        all_data = json.load(f)

    bvals = all_data.pop('config')
    bvals = bvals['bvalues']
    for name, data in all_data.items():
        yield name, bvals, data


def data_ivim_fit_saved(datafile, algorithmFile):
    # Find the algorithms from algorithms.json
    current_folder = pathlib.Path.cwd()
    algorithm_path = current_folder / algorithmFile
    with algorithm_path.open() as f:
        algorithm_information = json.load(f)
    # Load generic test data generated from the included phantom: phantoms/MR_XCAT_qMRI
    generic = current_folder / datafile
    with generic.open() as f:
        all_data = json.load(f)
    algorithms = algorithm_information["algorithms"]
    bvals = all_data.pop('config')
    bvals = bvals['bvalues']
    for algorithm in algorithms:
        first = True
        for name, data in all_data.items():
            algorithm_dict = algorithm_information.get(algorithm, {})
            xfail = {"xfail": name in algorithm_dict.get("xfail_names", {}),
                "strict": algorithm_dict.get("xfail_names", {}).get(name, True)}
            kwargs = algorithm_dict.get("options", {})
            tolerances = algorithm_dict.get("tolerances", {})
            skiptime=False
            if first:
                if algorithm_dict.get("fail_first_time", False):
                    skiptime = True
                    first = False
            requires_matlab = algorithm_dict.get("requires_matlab", False)
            yield name, bvals, data, algorithm, xfail, kwargs, tolerances, skiptime, requires_matlab

def algorithmlist(algorithmFile):
    # Find the algorithms from algorithms.json
    current_folder = pathlib.Path.cwd()
    algorithm_path = current_folder / algorithmFile
    with algorithm_path.open() as f:
        algorithm_information = json.load(f)

    algorithms = algorithm_information["algorithms"]
    for algorithm in algorithms:
        algorithm_dict = algorithm_information.get(algorithm, {})
        requires_matlab = algorithm_dict.get("requires_matlab", False)
        yield algorithm, requires_matlab

def bound_input(datafile,algorithmFile):
    # Find the algorithms from algorithms.json
    current_folder = pathlib.Path.cwd()
    algorithm_path = current_folder / algorithmFile
    with algorithm_path.open() as f:
        algorithm_information = json.load(f)
    # Load generic test data generated from the included phantom: phantoms/MR_XCAT_qMRI
    generic = current_folder / datafile
    with generic.open() as f:
        all_data = json.load(f)
    algorithms = algorithm_information["algorithms"]
    bvals = all_data.pop('config')
    bvals = bvals['bvalues']
    for name, data in all_data.items():
        for algorithm in algorithms:
            algorithm_dict = algorithm_information.get(algorithm, {})
            xfail = {"xfail": name in algorithm_dict.get("xfail_names", {}),
                "strict": algorithm_dict.get("xfail_names", {}).get(name, True)}
            kwargs = algorithm_dict.get("options", {})
            tolerances = algorithm_dict.get("tolerances", {})
            requires_matlab = algorithm_dict.get("requires_matlab", False)
            yield name, bvals, data, algorithm, xfail, kwargs, tolerances, requires_matlab


@pytest.fixture(scope="session")
def threeddata(request):
    current_folder = pathlib.Path.cwd()
    datafile = request.config.getoption("dataFile")
    generic = current_folder / datafile
    with generic.open() as f:
        all_data = json.load(f)
    bvals = all_data.pop('config')
    bvals = np.array(bvals['bvalues'])
    sig, _, Dim, fim, Dpim, _=phantom(bvals, 1/1000, TR=3000, TE=40, motion=False, rician=False, interleaved=False, T1T2=True)
    return sig[16::,16::,8::,:], Dim[16::,16::,8::], fim[16::,16::,8::], Dpim[16::,16::,8::], bvals


'''''@pytest.fixture
def pardat(request):
    datafile = request.config.getoption("dataFile")
    current_folder = pathlib.Path.cwd()
    generic = current_folder / datafile
    with generic.open() as f:
        all_data = json.load(f)
    bvals = all_data.pop('config')['bvalues']

    data_list = []
    for name, data in all_data.items():
        signal = np.asarray(data["data"])
        signal /= signal[0]
        data_list.append(signal)

    big_array = np.stack(data_list, axis=0)
    return big_array, bvals'''''