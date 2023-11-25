import unittest
from raanova.ANOVA.anova import ANOVA
import numpy as np


class TestANOVA(unittest.TestCase):
    def test_base(self):
        n = 10000
        p = 10
        X = np.zeros((n, p))
        Y = np.zeros(n)
        actual_avg = np.random.random(p)*10 - 5
        for i in range(n):
            rdm = np.random.randint(10)
            X[i, rdm] = 1
            Y[i] = actual_avg[rdm] + np.random.randn()

        anova = ANOVA()
        betas = anova.fit(X, Y)
        print(betas)
        print(actual_avg)
