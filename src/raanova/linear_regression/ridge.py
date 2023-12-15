import numpy as np
import numpy.typing as npt

from .linear_regression import LinearRegression
from .helper_functions import get_residuals, get_variance, get_r_squared
from .helper_functions import get_ridge_hat_ann_matrix, get_OLS_CI


class Ridge(LinearRegression):
    def __init__(self):
        super().__init__()
        self._penalty = 0

    def fit(self, X: list[list[float]], Y: list[float], intercept: bool = True, penalty: float = 0.1) -> list[float]:
        if intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))

        n, p = np.shape(X)
        self._betas = np.linalg.inv(X.T @ X + self._penalty * np.identity(p)) @ (X.T @ Y)

        self._residuals = get_residuals(X, Y, self._betas)
        self._sigma_naive, self._sigma_corrected = get_variance(n, p, self._residuals)
        self._rsquared = get_r_squared(n, Y, self._sigma_naive)
        self._hat, self._annihilator = get_ridge_hat_ann_matrix(X, self._penalty)

        return self._betas

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass
