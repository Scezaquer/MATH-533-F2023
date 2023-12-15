import unittest
from raanova.linear_regression.wls import WLS
import numpy as np


class TestWLS(unittest.TestCase):
    def test_1(self):
        X = np.random.uniform(-1,1,(100,5))
        e = np.random.normal(0, 0.1, (100,1))
        Y = (0.5 + 1.0*X[:,0] + 1.0*X[:,1]+1.0*X[:,2]).reshape(-1,1)+e

        #X = np.atleast_2d(X).T
        #Y = np.atleast_2d(Y).T

        wls = WLS()
        W = np.diag(np.full(len(X),1))
        beta = wls.fit(X, Y, W)

        print(f"betas:\n{wls._betas}")
        print(f"res: {wls._residuals[:10, :]}")
        print(f"r_sqrd: {wls._rsquared}")
        print(f"naive: {wls._sigma_naive}")
        print(f"corrected: {wls._sigma_corrected}")

        
        






