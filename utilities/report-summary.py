import pathlib
import json
import sys

def summarize_test_report(input_file:str, output_file:str):
    file = pathlib.Path(__file__)
    report_path = file.with_name(input_file)
    with report_path.open() as f:
        report_info = json.load(f)
    summary = []
    for test_case in report_info['tests']:
        values = test_case['user_properties'][0]['test_data']
        values['status'] = test_case['outcome']
        summary.append(values)

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python report-summary.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    summarize_test_report(input_file, output_file)
