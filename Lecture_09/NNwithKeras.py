import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_circles
import keras
from keras.utils import np_utils
from keras.layers import Dense, Activation
from keras.models import Sequential

X, y = make_circles(n_samples=1000, factor=0.4)
print X.shape, y.shape 

plt.scatter(X[:, 0], X[:, 1])
plt.show()


for ix in range(X.shape[0]):
    if y[ix] == 0:
        plt.scatter(X[ix, 0], X[ix, 1], color='red')
    else:
        plt.scatter(X[ix, 0], X[ix, 1], color='green')


Y = np_utils.to_categorical(y)
print Y.shape

for ix in range(10):
    print Y[ix], "---", y[ix]
