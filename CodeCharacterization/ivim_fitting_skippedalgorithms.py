import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
from utilities.data_simulation.Download_data import download_data
import json
from src.wrappers.OsipiBase import OsipiBase
import matlab.engine

SNR = 20
data_path = '../'
data_filename = rf"test_output_SNR{SNR}_complete_bounds.csv"
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
signal_cols = [c for c in single_dataset.columns if c.startswith("b")]
results_rows = []

for name in algorithmlist:
	print(f"Starting analysis for algorithm: {name}")
	for anatomy in single_dataset["Region"].unique():
		print(f"Working on region: {anatomy}")
		bvals = data_json[anatomy]["bvalues"]
		data_region = single_dataset[single_dataset["Region"] == anatomy]
		fit = OsipiBase(algorithm=name[0], bvalues=bvals)
		fit_results = []
		for i in range(len(data_region)):
			row = data_region.iloc[i].copy()
			row["Algorithm"] = name
			signals = data_region.iloc[i][signal_cols].values.astype(float)
			data_norm = signals[~np.isnan(signals)]
			maps = fit.osipi_fit(data_norm, bvals)
			row["f_fitted"] = maps["f"]
			row["Dp_fitted"] = maps["Dp"]
			row["D_fitted"] = maps["D"]
			results_rows.append(row)
df_fitted = pd.DataFrame(results_rows)
save_filename = rf"test_output_skippedalgorithms_SNR{SNR}_complete_bounds.csv"
df_fitted.to_csv(os.path.join(data_path,save_filename), index=False)