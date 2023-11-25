import numpy as np
import numpy.typing as npt


class LinearRegression:
    def __init__(self):
        self._residuals: npt.NDArray[np.float32] = 0
        self._rsquared: float = 0
        self._betas: npt.NDArray[np.float32] = 0
        self._conf_interval: npt.NDArray[np.float32] = 0
        self._sigma_naive: float = 0
        self._sigma_corrected: float = 0
        self._AIC: float = 0
        self._BIC: float = 0
        self._hat: npt.NDArray[np.float32] = 0
        self._annihilator: npt.NDArray[np.float32] = 0

    def predict(self, X: npt.NDArray[np.float32], intercept: bool = True
                ) -> npt.NDArray[np.float32]:
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X @ self._betas

    def summary(self) -> None:
        # TODO: Complete here
        print("Summary of the Linear regression")
        print(f"R squared:          {round(self._rsquared, 3):>10}")
        print(f"Sigma naive:        {round(self._sigma_naive, 3):>10}")
        print(f"Sigma corrected:    {round(self._sigma_corrected, 3):>10}")
        print(f"AIC:                {round(self._AIC, 3):>10}")
        print(f"BIC:                {round(self._BIC, 3):>10}")

    @property
    def residuals(self):
        return self._residuals

    @property
    def rsquared(self):
        return self._rsquared

    @property
    def betas(self):
        return self._betas

    @property
    def conf_interval(self):
        return self._conf_interval

    @property
    def sigma_naive(self):
        return self._sigma_naive

    @property
    def sigma_corrected(self):
        return self._sigma_corrected

    @property
    def AIC(self):
        return self._AIC

    @property
    def BIC(self):
        return self._BIC

    @property
    def hat(self):
        return self._hat

    @property
    def annihilator(self):
        return self._annihilator
