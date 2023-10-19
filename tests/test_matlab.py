import numpy as np
from tests.run_matlab import run

def test_run_matlab():
    x = 4.0
    np.testing.assert_array_equal(np.sqrt(x),run(x))