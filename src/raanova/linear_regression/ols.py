from .linear_regression import LinearRegression
import numpy as np
import numpy.typing as npt


class OLS(LinearRegression):
    def __init__(self):
        super().__init__()

    def fit(self,
            X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
            ) -> npt.NDArray[np.float32]:

        # Add a column of ones to the input data for the intercept term
        X = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate the coefficients using the normal equation
        self.__beta = np.linalg.inv(X.T @ X) @ X.T @ Y

        return self.__beta
