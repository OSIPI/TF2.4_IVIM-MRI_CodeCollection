import os
import pandas as pd
import json

# directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# path to the repository
REPO_DIR = os.path.dirname(SCRIPT_DIR)

CODE_CONTRIBUTIONS_FILE = os.path.join(REPO_DIR, "doc", "code_contributions_record.csv")
ALGORITHMS_FILE = os.path.join(REPO_DIR, "tests", "IVIMmodels", "unit_tests", "algorithms.json")
SOURCE_FOLDER = os.path.join(REPO_DIR, "src", "original")
WRAPPED_FOLDER = os.path.join(REPO_DIR, "src", "standardized")

# Read the CSV file
df = pd.read_csv(CODE_CONTRIBUTIONS_FILE)

unique_subfolders = df['subfolder'].unique().tolist()

# Read the JSON file
with open(ALGORITHMS_FILE, 'r') as f:
    algorithms_data = json.load(f)

# list of all algorithms from the JSON file
all_algorithms = algorithms_data['algorithms']

# Check if both code_contributions_file matches with source folder
for subfolder in unique_subfolders:
    subfolder_path = os.path.join(SOURCE_FOLDER, subfolder)
    if not os.path.exists(subfolder_path):
        print(f"Warning: Subfolder '{subfolder}' does not exist in the source folder.")

# Add column 'Tested' to the DataFrame based on a match with algorithms and wrapped column
df['Tested'] = df.apply(lambda row: 'Yes' if any(algorithm in row['wrapped'] for algorithm in all_algorithms) else 'No', axis=1)

# Select the desired columns
df_selected = df[['Technique', 'subfolder', 'Authors', 'Tested', 'wrapped']]
df_selected.columns = ['Technique', 'Subfolder', 'Contributors', 'Tested', 'wrapped']

# Convert the DataFrame to HTML
html_string = df_selected.to_html(index=False)

# Save the HTML to a file
with open(os.path.join(REPO_DIR, 'website','combined_report.html'), 'w') as f:
    f.write(html_string)

# Printing message that report has been successfully generated
print("Combined HTML report generated successfully.")