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
        
        self._n: int = 0  #sample size
        self._p: int = 0  #num of covs
        self._X: npt.NDArray[np.float32] = 0

    def fit(
        self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
        pass

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        #i added attributes n and p bc i need them for summary display
        self._n = len(X)
        self._p = len(X[0])

        #why is this printing something? 
        X = np.column_stack((np.ones(X.shape[0]), X))
        print(X.shape)
        print(self._betas.shape)
        return X @ self._betas


    def summary(self) -> None:
        #residuals section:
        #min/max/med/quartiles
        min_res = np.min(self._residuals)
        Q1 = np.percentile(self._residuals, 25)
        median = np.percentile(self._residuals,50)
        Q3 = np.percentile(self._residuals,75)
        max_res = np.max(self._residuals)
        
        #coefficients section:
        res_std_error_squared = sum(self._residuals**2)/(self._n-self._p-1)
        beta_std_errors = np.atleast_2d((self._sigma_corrected % np.lingalg.inv(self._X.T % self._X))**(1/2))
        t_val = np.atleast_2d(np.divide(self._betas, beta_std_errors))
        #missing P(>|t|) and F stat
        
        res_std_error = (res_std_error_squared)**(1/2)
        
        print(f"{'Residuals:' : <10}\n"
              f"{'Min': <10}{'Q1': ^10}{'Med': ^10}{'Q3': ^10}{'Max': >5}"
              f"\n{min_res: <10}{Q1: ^10}{median: ^10}{Q3: ^10}{max_res: >5}\n\n")
        print(f"{'Coefficients:' : <10}\n"
              f"{'Estimate': <10}{'Std. Error': ^10}{' t value': >5}")
        for a,b,c in zip(self._betas, beta_std_errors, t_val):
            print(f"{a[0]: <10}{b[0]: ^10}{c[0]: >5}")
        print('\n\n'+'Residual standard error: ' + str(res_std_error[0]) 
              + '\n R-squared: ' + str(self._rsquared))
        
    #obviously didnt add a getter for the betas since theyre returned by the fit() fcn
    @property
    def residuals(self):
        return self._residuals
    
    @property
    def rsquared(self):
        return self._rsquared
    
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
    
    
    
    
    
    
    
    
    
    
