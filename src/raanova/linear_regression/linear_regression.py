import numpy as np
import numpy.typing as npt


class LinearRegression:
    def __init__(self):
        self.__residuals: npt.NDArray[np.float32] = 0
        self.__rsquared: float = 0
        self.__betas: npt.NDArray[np.float32] = 0
        self.__conf_interval: npt.NDArray[np.float32] = 0
        self.__sigma_naive: float = 0
        self.__sigma_corrected: float = 0
        self.__AIC = 0
        self.__BIC = 0
        self.__hat: npt.NDArray[np.float32] = 0
        self.__annihilator = 0

    def fit(
        self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
        pass

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

    def summary(self) -> None:
        pass

    @property
    def residuals(self):
        return self.__residuals
