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

