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
        self._AIC = 0
        self._BIC = 0
        self._hat: npt.NDArray[np.float32] = 0
        self._annihilator = 0

    def fit(
        self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
        pass

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        X = np.column_stack((np.ones(X.shape[0]), X))
        print(X.shape)
        print(self._betas.shape)
        return X @ self._betas

    def summary(self) -> None:
        pass

    @property
    def residuals(self):
        return self._residuals
