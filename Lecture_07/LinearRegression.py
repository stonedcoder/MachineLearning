import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=1, bias=4.2,noise=5.1)
print X.shape, y.shape

split = int(0.8*X.shape[0])
print split

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print X_train.shape
print X_test.shape

print y_test.shape

## Methods we need to write
def hypothesis(x,w):
    x0 = 1
    return w[0]*x0 + x*[1]