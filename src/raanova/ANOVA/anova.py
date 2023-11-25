import numpy as np
import numpy.typing as npt


class ANOVA:
    def __init__(self):
        self._f_statistic = 0
        self._p_value = 0
        self._df = 0
        self._bss = 0
        self._tss = 0
        self._wss = 0
        self._n = 0
        self._g = 0
        self._ybar = 0

    def fit(self,
            X: npt.NDArray[np.int32], Y: npt.NDArray[np.float32],
            intercept: bool = False) -> npt.NDArray[np.float32]:

        # TODO: update the attributes
        # TODO: check intercept works
        # TODO: add guards

        # Add a column of ones to the input data for the intercept term
        if intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate the coefficients using the normal equation
        self._betas = np.sum(X*Y[:, np.newaxis], 0)/np.sum(X, 0)
        self._ybar = np.average(Y)

        self._tss = sum([(y - self._ybar)**2 for y in Y])
        self._wss = sum([])

        return self._betas

    def print_table(self):
        pass
