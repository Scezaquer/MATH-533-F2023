import numpy as np
import numpy.typing as npt

from .linear_regression import LinearRegression
from .helper_functions import *


class Ridge(LinearRegression):
    def __init__(self, penalty: float = 0.1, intercept: bool = True):
        super().__init__()
        self._penalty = penalty
        

    def fit(self, X: list[list[float]], Y: list[float], intercept: bool = True) -> list[float]:
        if intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))

        n, p = np.shape(X)
        self._betas = np.linalg.inv(X.T @ X + self._penalty * np.identity(p)) @ (X.T @ Y)

        self._residuals = get_residuals(X, Y, self._betas)
        self._sigma_naive, self._sigma_corrected = get_variance(n, p, self._residuals)
        self._rsquared = get_r_squared(n, Y, self._sigma_naive)
        self._conf_interval = get_OLS_CI(self._betas, self._sigma_corrected, X, n, p, alpha= 0.05)
        self._hat, self._annihilator = get_hat_ann_matrix(X)
        
        return self._betas

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

