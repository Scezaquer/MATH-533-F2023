from .linear_regression import LinearRegression
from .mila_helpers import *
import numpy as np



class WLS(LinearRegression):
    
    def __init__(self):
        super().__init__()

    def fit(self, X: list[list[float]], Y: list[float], W: list[list[float]], intercept: bool = True) -> list[float]:
        
        if intercept == True:
            #add col for intercept
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        #define default weight matrix if none entered
        #commented whole thing out bc we are not doing this in here
        """if W == None:
            #to get default W, need var of sample residuals
            
            #need to calc variance sample residuals
            #so we get ols beta_hats first
            OLS_beta = np.atleast_2d(np.linalg.inv(X.T @ X) @ X.T @ Y)
            self._betas = OLS_beta
            
            sample_res = Y-super().predict(Y)
            
            #calc sample var
            #mean = np.sum(sample_res)/len(sample_res)
            #var = np.array(len(sample_res))
            #for e in sample_res:
            #    var += (e-mean)**2
            #var = var/len(sample_res)
            
            
            #set W matrix with diag 1/var
            W = np.diag(np.full(len(X),1))
            W.dtype = np.float32
            for d in range(0,len(W)):
                W[d][d] = 1/(sample_res[len(X[0])-1][d]**2)
        """     
        b = np.atleast_2d(np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y)
        res = Y - (X @ b)
        
        n = len(X)
        p = len(X[0])
        
        #get sigma naive for r^2
        var = get_variance(n, p, res)
        #hat, ann matrices
        h_a = get_hat_ann_matrix(X)
        
        #aic, bic
        #aic_bic = get_OLS_AIC_BIC(Y, super().predict(Y), n, p)
        
        self._betas = b
        self._residuals = res
        self._rsquared = get_r_squared(n, Y, var[0])
        #self._conf_interval = get_OLS_CI(b, var[1], X, n, p)
        self._sigma_naive = var[0]
        self._sigma_corrected = var[1]
        #self._AIC = aic_bic[0]
        #self._BIC = aic_bic[1]
        self._hat = h_a[0]
        self._annihilator = h_a[1]
        
        
        return b
        
        
