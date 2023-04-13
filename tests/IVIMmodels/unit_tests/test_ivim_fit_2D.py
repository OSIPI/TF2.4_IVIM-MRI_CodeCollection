import numpy as np


class TestIVIMFit2D:

    def __init__(self):
        self.b_values = None
        self.signals = None
        self.supervision = {'D': None,
                            'Dp': None,
                            'f': None,
                            'S0': None}

    @staticmethod
    def estimation_method(b_values, signals):
        D = 0.003
        Dp = 0.02
        f = 0.3
        S0 = 1
        return D, Dp, f, S0

    def test_ivim_fit_2D(self):
        D, Dp, f, S0 = self.estimation_method(self.b_values, self.signals)
        assert np.allclose(D, self.supervision['D'])
        assert np.allclose(Dp, self.supervision['Dp'])
        assert np.allclose(f, self.supervision['f'])
        assert np.allclose(S0, self.supervision['S0'])

