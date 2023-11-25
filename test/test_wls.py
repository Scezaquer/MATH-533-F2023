import unittest
from raanova.linear_regression.wls import WLS
import numpy as np


class TestWLS(unittest.TestCase):
    def test_1(self):
        X = np.random.random(size=1000)*100
        e = np.random.normal(0, 1, size=1000)
        Y = 2.5*X + e

        X = np.atleast_2d(X).T
        Y = np.atleast_2d(Y).T

        wls = WLS()
        W = np.diag(np.full(len(X),1))
        beta = wls.fit(X, Y, W)

        print(beta)

        X = np.random.random(size=10)*100
        y_hat = wls.predict(X)
        print(y_hat)
    def test_2(self):
        X = np.random.random(size=1000)*100
        e = np.random.normal(0, 1, size=1000)
        Y = 2.5*X + e
        
        X = np.atleast_2d(X).T
        Y = np.atleast_2d(Y).T
        
        wls = WLS()
        W = np.diag(np.full(len(X),1))
        W.dtype = np.float32
        w = 0.1
        for d in range(0,len(W)):
            W[d][d] = w
            w += 0.2
        beta = wls.fit(X, Y, W)

        print(beta)

        X = np.random.random(size=10)*100
        y_hat = wls.predict(X)
        print(y_hat)
        
        






