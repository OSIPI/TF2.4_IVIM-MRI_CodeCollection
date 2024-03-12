# 4D IVIM Phantom Generator

A command-line tool for generating a 4D IVIM (Intravoxel Incoherent Motion) phantom as a NIfTI file.

## Usage

```sh
python sim_vim_sig.py  [-h] [-b BVALUE [BVALUE ...] | -f FILE] [-n NOISE] [-m] [-i]
```

## Arguments

- `-b`, `--bvalue` : B values (list of numbers)
- `-f`, `--bvalues-file` : JSON file containing the B values (default: b_values.json)
- `-n`, `--noise` : Noise level (default: 0.0005)
- `-m`, `--motion` : Enable motion flag (default: False)
- `-i`, `--interleaved` : Enable interleaved flag (default: False)

**Note:** Either provide `--bvalue` or `--bvalues-file`, not both.

If neither `--bvalues-file` nor `--bvalue` is provided, the script will use the default `b_values.json` file.

## Customizing B Values

You can customize the B values by editing the `b_values.json` file. Here's an example of the default format:

```json
{
    "original": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 350.0, 400.0, 550.0, 700.0, 850.0, 1000.0],
    "one": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 350.0, 400.0, 550.0, 700.0, 850.0, 1000.0, 1100.0, 1200.0],
    "two": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 350.0, 400.0, 500.0, 700.0, 800.0, 1000.0, 1100.0, 1200.0],
    "three": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 350.0, 450.0, 550.0, 675.0, 800.0],
    "four": [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
}
```

You can add or remove values as needed for your simulation. Here are the effects of customizing the B values for each of the generic profiles (`generic.json`, `generic_one.json`, `generic_two.json`, `generic_three.json`, `generic_four.json`):

The default `generic_<custom_name>.json` file can be found in the following location:

```
TF2.4_IVIM-MRI_CodeCollection/tests/IVIMmodels/unit_tests
```

## Running the Script

To run the script, you can add the following line to your shell configuration file (e.g., `.bashrc` or `.zshrc`) to include the necessary directory in your `PYTHONPATH`:

```sh
export PYTHONPATH=$PYTHONPATH:~/TF2.4_IVIM-MRI_CodeCollection
```

Replace `~/TF2.4_IVIM-MRI_CodeCollection` with the actual path to the directory containing the script.

## Example

```sh
python sim_ivim_sig.py -n 0.0001 -m
```

This command will generate a 4D IVIM phantom using B values from the default `b_values.json` file, with a noise level of 0.0001 and motion enabled.
