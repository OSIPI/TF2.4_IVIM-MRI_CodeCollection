import os
import pandas as pd
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv(r'C:\Users\home\tf2.4\TF2.4_IVIM-MRI_CodeCollection\local.env')

# Read the CSV file
csv_file_path = os.getenv('CODE_CONTRIBUTIONS_FILE')
df = pd.read_csv(csv_file_path)

unique_subfolders = df['subfolder'].unique().tolist()

# Read the JSON file
algorithms_file_path = os.getenv('ALGORITHMS_FILE')
with open(algorithms_file_path, 'r') as f:
    algorithms_data = json.load(f)

# Get a list of all algorithms from the JSON file
all_algorithms = algorithms_data['algorithms']

# Add a new column 'Tested' to the DataFrame if it starts with that of subfolder
df['Tested'] = df.apply(lambda row: 'Yes' if any(algorithm.startswith(row['subfolder'].split('_')[0]) for algorithm in all_algorithms) else 'No', axis=1)

# Select the desired columns
df_selected = df[['Technique', 'subfolder', 'Authors', 'Tested']]
df_selected.columns = ['Technique', 'Subfolder', 'Contributors', 'Tested']

# Convert the DataFrame to HTML
html_string = df_selected.to_html(index=False)

# Save the HTML to a file
with open('combined_report.html', 'w') as f:
    f.write(html_string)
# printing message that report have been succesfully generated
print("Combined HTML report generated successfully.")