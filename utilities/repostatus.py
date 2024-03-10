import os
import pandas as pd
import json

# directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# path to the repository
REPO_DIR = os.path.dirname(script_dir)

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

# Add column 'Tested' to the DataFrame if it starts with that of subfolder
df['Tested'] = df.apply(lambda row: 'Yes' if any(algorithm.startswith(row['subfolder'].split('_')[0]) for algorithm in all_algorithms) else 'No', axis=1)

# Parse files in the WRAPPED_FOLDER 
wrapped_algorithms = []
for root, dirs, files in os.walk(WRAPPED_FOLDER):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                content = f.read()
                for algorithm in all_algorithms:
                    if algorithm in content:
                        wrapped_algorithms.append(algorithm)

# Add a column 'Wrapped' to the DataFrame
df['Wrapped'] = df.apply(lambda row: 'Yes' if any(algorithm.startswith(row['subfolder'].split('_')[0]) for algorithm in wrapped_algorithms) else 'No', axis=1)

# Select the desired columns
df_selected = df[['Technique', 'subfolder', 'Authors', 'Tested', 'Wrapped']]
df_selected.columns = ['Technique', 'Subfolder', 'Contributors', 'Tested', 'Wrapped']

# Convert the DataFrame to HTML
html_string = df_selected.to_html(index=False)

# Save the HTML to a file
with open(os.path.join(REPO_DIR, 'combined_report.html'), 'w') as f:
    f.write(html_string)

# Printing message that report has been successfully generated
print("Combined HTML report generated successfully.")