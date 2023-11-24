import unittest
from raanova.linear_regression.ols import OLS
import numpy as np


class TestOLS(unittest.TestCase):
    def test_base(self):
        X = np.random.random(size=1000)*100
        e = np.random.normal(0, 1, size=1000)
        Y = 2.5*X + e

        X = np.atleast_2d(X).T
        Y = np.atleast_2d(Y).T

        ols = OLS()
        beta = ols.fit(X, Y)

        print(beta)

        X = np.random.random(size=10)*100
        y_hat = ols.predict(X)
        print(y_hat)
        
        ols.summary()
