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

split = int(0.8*X.shape[0])
X_train = X[:split,:]
X_test = X[split:,:]

y_train = Y[:split]
y_test = Y[split:]

print X_train.shape


## NN in keras
model = Sequential()

model.add(Dense(5,input_shape = (2,)))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model2 = Sequential()

model2.add(Dense(2, input_shape=(2,)))
for ix in range(5):
    model2.add(Dense(5))
model2.summary()