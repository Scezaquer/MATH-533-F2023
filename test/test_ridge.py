import unittest
from raanova.linear_regression.ridge import Ridge
import numpy as np


class TestRidge(unittest.TestCase):
    def test_base(self):
        # X = np.random.random(size=1000)*1000
        # e = np.random.normal(0, X)
        # Y = X + e

        X = np.linspace(0.5, 10, 100)
        e = np.random.normal(0, 0.5, X.shape[0])
        Y = X + e

        X = np.atleast_2d(X).T
        Y = np.atleast_2d(Y).T

        m1 = Ridge(0.1)
        betas = m1.fit(X, Y)
        print("Ridge:")
        # print(f"betas: {m1._betas}")
        # print(f"residuals: {m1._residuals[:10, :]}")
        # print(f"r_squared: {m1._rsquared}")
        # print(f"sigma_naive: {m1._sigma_naive}")
        # print(f"sigma_corr: {m1._sigma_corrected}")
        # print(f"CI: {m1._conf_interval}")
        print("CI: ")
        m1._conf_interval
        # print(f"hat_mtx: {m1._hat}")
        # print(f"annihilator_mtx: {m1._annihilator}")
