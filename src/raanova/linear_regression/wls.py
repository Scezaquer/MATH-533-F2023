from .linear_regression import LinearRegression
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
        beta_hat = np.atleast_2d(np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y)
        self._betas = beta_hat
        
        return beta_hat
        
        
