import numpy as np
import run_matlab

def test_run_matlab():
    x = 4.0
    np.testing.assert_array_equal(np.sqrt(x),run_matlab.run(x))