# Imports
import numpy as np
from raanova import WLS

# Generate the dataset to perform regression on
# the true model is y = 0*1 + 1*x0 + 2*x1 + 4*x2
X = np.random.rand(200, 3) * 100
true_beta = np.atleast_2d([1, 2, 4]).T
epsilon = np.random.randn(200, 1)
Y = X @ true_beta + epsilon
W = np.diag(np.full(len(X), 1))

# Fit
model = WLS()
beta_hat = model.fit(X, Y, W, intercept=False)

# Display model information
model.summary()

# Make predictions on new data
X = np.random.rand(10, 3) * 100
y_hat = model.predict(X, intercept=False)
print(y_hat)
