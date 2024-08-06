import pandas as pd

df = pd.read_csv('test_output.csv')

# Columns to be rounded to four decimal places
columns_to_round = [
    'f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted',
    'bval_0.0', 'bval_1.0', 'bval_2.0', 'bval_5.0', 'bval_10.0', 'bval_20.0',
    'bval_30.0', 'bval_50.0', 'bval_75.0', 'bval_100.0', 'bval_150.0', 'bval_250.0',
    'bval_350.0', 'bval_400.0', 'bval_550.0', 'bval_700.0', 'bval_850.0', 'bval_1000.0'
]
for column in columns_to_round:
    df[column] = df[column].round(4)

#df = df.loc[:, ~df.columns.str.startswith('bval')] 

#compress and save the file.
df.to_csv('test_output.csv.gz', compression='gzip', index=False)
