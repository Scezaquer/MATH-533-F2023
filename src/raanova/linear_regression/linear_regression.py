import numpy as np


class LinearRegression:
    def __init__(self):
        self.__residuals: np.ndarray[float] = 0
        self.__rsquared: float = 0
        self.__betas: np.ndarray[float] = 0
        self.__conf_interval: np.ndarray[float] = 0
        self.__sigma_naive: float = 0
        self.__sigma_corrected: float = 0
        self.__AIC = 0
        self.__BIC = 0
        self.__hat: np.ndarray[float] = 0
        self.__annihilator = 0

    def fit(
        self, X: np.ndarray[float], Y: np.ndarray[float]
        ) -> np.ndarray[float]:
        pass

    def predict(self, X: np.ndarray[float]) -> np.ndarray[float]:
        pass

    def summary(self) -> None:
        pass

    @property
    def residuals(self):
        return self.__residuals
