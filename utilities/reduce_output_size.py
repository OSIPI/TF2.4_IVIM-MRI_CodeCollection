import pandas as pd
import os

file_path = 'test_output.csv'

# Check if the file exists
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Columns to be rounded to four decimal places
    columns_to_round = ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']
    for column in columns_to_round:
        df[column] = df[column].round(4)

    #drop b_values columns.
    df = df.loc[:, ~df.columns.str.startswith('bval')] 

    #compress and save the file.
    df.to_csv('test_output.csv.gz', compression='gzip', index=False)
