import os
import gzip
import csv

file_path = 'test_output.csv'

# Check if the file exists
if os.path.exists(file_path):
    # Open the input and output files
    with open(file_path, 'r') as infile, gzip.open('test_output.csv.gz', 'wt', newline='') as outfile:
        reader = csv.DictReader(infile)

        # Drop b_values columns
        fieldnames = [field for field in reader.fieldnames if not field.startswith('bval_')]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        columns_to_round = ['f', 'Dp', 'D', 'f_fitted', 'Dp_fitted', 'D_fitted']
        
        # Process each row
        for row in reader:
            filtered_row = {column: row[column] for column in fieldnames}
            for column in columns_to_round:
                if column in filtered_row:
                    filtered_row[column] = round(float(filtered_row[column]), 4)
            writer.writerow(filtered_row)
