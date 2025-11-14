import yaml
import pytest
import numpy as np
from pathlib import Path
import json
from scipy import stats

# TODO: These should probably be fixtures
from src.wrappers.OsipiBase import OsipiBase
from utilities.data_simulation.GenerateData import GenerateData


def load_config(path):
    """Loads a YAML configuration file."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, path):
    """Saves a YAML configuration file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def get_algorithms():
    """Loads the list of algorithms from the JSON file."""
    algorithms_path = Path(__file__).parent / "algorithms.json"
    with open(algorithms_path, "r") as f:
        return json.load(f)["algorithms"]

def generate_config_for_algorithm(algorithm):
    """Generates reference data for a given algorithm."""
    fit_count = 300
    snr = 100
    rician_noise = True
    
    regions = {
        "Blood RV": {"f": 1.0, "Dp": 0.1, "D": 0.003},
        "Myocardium LV": {"f": 0.15, "Dp": 0.08, "D": 0.0024},
        "myocardium RV": {"f": 0.15, "Dp": 0.08, "D": 0.0024},
        "myocardium ra": {"f": 0.07, "Dp": 0.07, "D": 0.0015},
    }
    bvals = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800])

    new_config_entry = {}

    print(f"Generating reference data for {algorithm}")
    
    if "NET" in algorithm or "DC" in algorithm or "MATLAB" in algorithm:
        print("  Skipping deep learning or MATLAB algorithm")
        return None

    new_config_entry[algorithm] = {}
    for region_name, data in regions.items():
        print(f"  Running {algorithm} for {region_name}")
        
        rng = np.random.RandomState(42)
        S0 = 1
        gd = GenerateData(rng=rng)
        D = data["D"]
        f = data["f"]
        Dp = data["Dp"]
        
        try:
            fit = OsipiBase(algorithm=algorithm)
        except Exception as e:
            print(f"    Could not instantiate {algorithm}: {e}")
            continue

        results = {"f": [], "Dp": [], "D": []}
        for idx in range(fit_count):
            signal = gd.ivim_signal(D, Dp, f, S0, bvals, snr, rician_noise)
            try:
                fit_result = fit.osipi_fit(signal, bvals)
                results["f"].append(fit_result["f"])
                results["Dp"].append(fit_result["Dp"])
                results["D"].append(fit_result["D"])
            except Exception as e:
                print(f"    Fit failed for {algorithm} at index {idx}: {e}")

        if results["f"]:
            f_mu = float(np.mean(results["f"]))
            Dp_mu = float(np.mean(results["Dp"]))
            D_mu = float(np.mean(results["D"]))
            f_std = float(np.std(results["f"]))
            Dp_std = float(np.std(results["Dp"]))
            D_std = float(np.std(results["D"]))
            
            new_config_entry[algorithm][region_name] = {
                100: {
                    "ground_truth": {"f": float(f), "Dp": float(Dp), "D": float(D)},
                    "acceptance_criteria": {
                        "f": {"mean": f_mu, "std_dev": f_std, "mean_tolerance": 0.2, "std_dev_tolerance_percent": 100.0},
                        "Dp": {"mean": Dp_mu, "std_dev": Dp_std, "mean_tolerance": 0.2, "std_dev_tolerance_percent": 100.0},
                        "D": {"mean": D_mu, "std_dev": D_std, "mean_tolerance": 0.002, "std_dev_tolerance_percent": 100.0},
                    }
                }
            }
    return new_config_entry

def run_simulation_batch(algorithm, bvals, ground_truth, snr, batch_size, rician_noise):
    """Runs a batch of simulations and returns the fitted parameters."""
    fit = OsipiBase(algorithm=algorithm)
    rng = np.random.RandomState()
    gd = GenerateData(rng=rng)
    S0 = 1

    results = {"f": [], "Dp": [], "D": []}

    for _ in range(batch_size):
        signal = gd.ivim_signal(ground_truth["D"], ground_truth["Dp"], ground_truth["f"], S0, bvals, snr, rician_noise)
        fit_result = fit.osipi_fit(signal, bvals)
        results["f"].append(fit_result["f"])
        results["Dp"].append(fit_result["Dp"])
        results["D"].append(fit_result["D"])
    return results


@pytest.mark.parametrize("algorithm", get_algorithms())
def test_statistical_equivalence(algorithm):
    """
    Main test function to check statistical equivalence.
    """
    config_path = Path(__file__).parent / "statistical_config.yml"
    config = load_config(config_path)
    
    if algorithm not in config:
        proposed_config_path = Path(__file__).parent / "proposed_statistical_config.yml"
        proposed_config = load_config(proposed_config_path)
        
        new_entry = generate_config_for_algorithm(algorithm)
        if new_entry:
            proposed_config.update(new_entry)
            save_config(proposed_config, proposed_config_path)
        
        pytest.fail(f"Algorithm {algorithm} not in statistical_config.yml. A new entry has been proposed in proposed_statistical_config.yml.")

    test_cases = config[algorithm]

    # B-values for simulation - this might need to be adjusted based on real data
    bvals = np.array([0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800])
    rician_noise = True
    batch_size = 25
    max_repetitions = 400 # 16 batches of 25
    alpha = 0.05 # For confidence intervals

    print(f"Running tests for algorithm: {algorithm}")

    all_tests_passed = True
    for region, snr_configs in test_cases.items():
        for snr, case_config in snr_configs.items():
            print(f"  Testing Region: {region}, SNR: {snr}")

            ground_truth = case_config["ground_truth"]
            acceptance_criteria = case_config["acceptance_criteria"]

            all_results = {"f": [], "Dp": [], "D": []}
            
            test_passed = False
            for i in range(max_repetitions // batch_size):
                print(f"    Batch {i+1}")
                batch_results = run_simulation_batch(algorithm, bvals, ground_truth, snr, batch_size, rician_noise)

                for param in all_results.keys():
                    all_results[param].extend(batch_results[param])

                # Update running statistics and check for early stopping
                passed_criteria = 0
                for param, criteria in acceptance_criteria.items():
                    values = all_results[param]
                    n = len(values)
                    if n < 2:
                        continue
                    
                    mean = np.mean(values)
                    std_dev = np.std(values, ddof=1)
                    
                    # Confidence interval for the mean
                    mean_ci = stats.t.interval(1 - alpha, n - 1, loc=mean, scale=stats.sem(values))
                    
                    # Check if CI is within tolerance
                    if (mean_ci[0] > criteria["mean"] - criteria["mean_tolerance"] and
                        mean_ci[1] < criteria["mean"] + criteria["mean_tolerance"]):
                        
                        # Check std dev tolerance
                        std_dev_tolerance = criteria["std_dev"] * (criteria["std_dev_tolerance_percent"] / 100.0)
                        if abs(std_dev - criteria["std_dev"]) < std_dev_tolerance:
                            passed_criteria += 1

                if passed_criteria == len(acceptance_criteria):
                    print("      All criteria met, stopping early.")
                    test_passed = True
                    break
            
            if not test_passed:
                print(f"      Test failed for Region: {region}, SNR: {snr}")
                all_tests_passed = False

    assert all_tests_passed, "One or more statistical equivalence tests failed."
