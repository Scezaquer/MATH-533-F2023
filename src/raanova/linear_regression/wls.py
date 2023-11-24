from linear_regression import LinearRegression
import numpy as np

class WLS(LinearRegression):
    
    def __init__(self):
        super().__init__()

    def fit(self, X: list[list[float]], Y: list[float], W: list[list[float]] = None) -> list[float]:
        #define default weight matrix if none entered
        if W == None:
            #to get default W, need var of sample residuals
            
            #need to calc variance sample residuals
            #so we get ols beta_hats first
            OLS_beta = np.linalg.inv(X.T @ X) @ Y
            self.__betas = OLS_beta
            
            sample_res = Y-super.predict(Y)
            
            #calc sample var
            mean = sum(sample_res)/len(sample_res)
            var = 0
            for e in sample_res:
                var += (e-mean)**2
            var = var/(len(sample_res)-1)
            
            #set W matrix with diag 1/var
            W = np.diag(np.full(len(X),1))
            for d in range(0,len(W)):
                W[d][d] = 1/var[d]
                
        beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y
        self.__betas = beta_hat
        return beta_hat
        
        
