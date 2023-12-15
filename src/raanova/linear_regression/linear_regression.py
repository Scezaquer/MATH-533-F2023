import numpy as np
import numpy.typing as npt
from scipy.stats import t
from .mila_helpers import *


class LinearRegression:
    def __init__(self):
        self._residuals: npt.NDArray[np.float32] = 0
        self._rsquared: float = 0
        self._betas: npt.NDArray[np.float32] = 0
        self._sigma_naive: float = 0
        self._sigma_corrected: float = 0
        
        
        #self._n: int = 0  #sample size
        #self._p: int = 0  #num of covs
        #self._X: npt.NDArray[np.float32] = 0

    def fit(
        self, X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
        pass

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        #i added attributes n and p bc i need them for summary display
        #self._n = len(X)
        #self._p = len(X[0])
        #self._X = X

        #why is this printing something? 
        X = np.column_stack((np.ones(X.shape[0]), X))
        #print(X.shape)
        #print(self._betas.shape)
        return X @ self._betas


    def summary(self) -> None:
        #residuals section:
        #min/max/med/quartiles
        min_res = round(np.min(self._residuals), 5)
        Q1 = round(np.percentile(self._residuals, 25), 5)
        median = round(np.percentile(self._residuals,50), 5)
        Q3 = round(np.percentile(self._residuals,75), 5)
        max_res = round(np.max(self._residuals), 5)
        
        #coefficients section:
        #res_std_error_squared = sum(self._residuals**2)/(self._n-self._p-1)
        #beta_std_errors = np.atleast_2d((self._sigma_corrected % np.linalg.inv(self._X.T % self._X))**(1/2))
        #t_val = np.atleast_2d(np.divide(self._betas, beta_std_errors))
        #p_values = t.sf(abs(t.val), df=self._n-self._p-1)*2
        
        #res_std_error = (res_std_error_squared)**(1/2)
        
        beta_i = []
        for i in range(0, len(self._betas)):
            beta_i.append('beta_'+str(i))
        
        
        print(f"{'Residuals:' : <10}\n"
              f"{'Min': <10}{'Q1': ^10}{'Med': ^10}{'Q3': ^10}{'Max': >5}"
              f"\n{min_res: <10}{Q1: ^10}{median: ^10}{Q3: ^10}{max_res: >5}\n\n")
        print('Coefficients      Estimates')
        
        for a, b in zip(beta_i, self._betas):
            print(f"{a: <10}        {round(b[0], 5): <10}")
        print()
        #for a,b,c,d in zip(self._betas, beta_std_errors, t_val, p_values):
            #print(f"{a[0]: <10}{b[0]: ^10}{c[0]: ^10}{d[0]: >5}")
        #print('\n\n'+'Residual standard error: ' + str(res_std_error[0]) 
        #      + '\n R-squared: ' + str(self._rsquared))
        
        CI_pretty_print(self._conf_interval, len(self._betas))
        print('\n\nR-squared: ' + str(round(self._rsquared, 5)) 
              +'\nNaive estimator: ' + str(round(self._sigma_naive, 5))
              + '\nCorrected naive estimator: ' + str(round(self._sigma_corrected, 5)))
        
    #obviously didnt add a getter for the betas since theyre returned by the fit() fcn
    @property
    def residuals(self):
        return self._residuals
    
    @property
    def rsquared(self):
        return self._rsquared
    
    @property
    def sigma_naive(self):
        return self._sigma_naive
    
    @property
    def sigma_corrected(self):
        return self._sigma_corrected
    
    
    
    
    
    
    
    
    
    
    
