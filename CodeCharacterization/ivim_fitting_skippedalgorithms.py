import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
from utilities.data_simulation.Download_data import download_data
import json
from src.wrappers.OsipiBase import OsipiBase
import matlab.engine
import ast

SNR = 20
data_path = '../'
data_filename = rf"test_output_SNR{SNR}_complete_bounds_initialguess_fullbvalues.csv"
data = pd.read_csv(os.path.join(data_path, data_filename))
data = data.dropna(how="all")
single_algorithm = data['Algorithm'].iloc[0]
single_dataset = data[data['Algorithm'] == single_algorithm]
json_path = '../tests/IVIMmodels/unit_tests/generic.json'
with open(json_path, "r") as f:
	data_json = json.load(f)

algorithms_path = '../tests/IVIMmodels/unit_tests/algorithms_skipped.json'

with open(algorithms_path, "r") as f:
	config = json.load(f)

algorithmlist = []

for name in config["algorithms"]:
	settings = config.get(name, {})
	requires_matlab = settings.get("requires_matlab", False)
	deep_learning = settings.get("deep_learning", False)
	algorithmlist.append((name, requires_matlab, deep_learning))

eng = matlab.engine.start_matlab()
results_rows = []

for name in algorithmlist:
	print(f"Starting analysis for algorithm: {name}")
	for anatomy in single_dataset["Region"].unique():
		print(f"Working on region: {anatomy}")
		bvals = data_json[anatomy]["bvalues_full"]
		data_region = single_dataset[single_dataset["Region"] == anatomy]

		initial_guess_f = np.fromstring(single_dataset["f_initial_guess"][0])[0]
		initial_guess_D = np.fromstring(single_dataset["D_initial_guess"][0])[0]
		initial_guess_Dp = np.fromstring(single_dataset["Dp_initial_guess"][0])[0]
		initial_guess = {
            "D": initial_guess_D,
            "f": initial_guess_f,
            "Dp": initial_guess_f,
            "S0": 1.0}

		bounds_f = np.fromstring(single_dataset["f_bounds"][0].strip("[]"), sep=", ")
		bounds_D = np.fromstring(single_dataset["D_bounds"][0].strip("[]"), sep=", ")
		bounds_Dp = np.fromstring(single_dataset["Dp_bounds"][0].strip("[]"), sep=", ")
		bounds = {
			"D": bounds_D,
			"f": bounds_f,
			"Dp": bounds_Dp,
			"S0": [0.5, 1.5]}

		fit = OsipiBase(algorithm=name[0], bvalues=bvals, initial_guess=initial_guess, bounds=bounds)
		fit_results = []
		for i in range(len(data_region)):
			row = data_region.iloc[i].copy()
			row["Algorithm"] = name
			signals = data_region.iloc[i]["measured_signals"]
			signals = np.fromstring(signals.strip("[]"), sep=" ")
			maps = fit.osipi_fit(signals, bvals, initial_guess=initial_guess, bounds=bounds)
			row["f_fitted"] = maps["f"]
			row["Dp_fitted"] = maps["Dp"]
			row["D_fitted"] = maps["D"]
			results_rows.append(row)
df_fitted = pd.DataFrame(results_rows)
save_filename = rf"test_output_skippedalgorithms_SNR{SNR}_complete_bounds_initialguess.csv"
df_fitted.to_csv(os.path.join(data_path,save_filename), index=False)