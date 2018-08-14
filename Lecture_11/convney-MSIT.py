import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,Dense,Activation,Dropout,Flatten, MaxPooling2D


import pandas as pd

ds = pd.read_csv('../../mnist_train.csv')
data = ds.values[1000:4500]
print data.shape

X = data[:, 1:]
y = data[:, 0]
Y = np_utils.to_categorical(y)
print np.unique(y)
print Y.shape



model = Sequential()
## Convolution Block 1
model.add(Convolution2D(32,3,3,input_shape=(28,28,1)))
model.add(Activation('relu'))

##Convolution Block 2 -> (b,26,26,32)
model.add(Convolution2D(16, 3,3))
model.add(Activation('relu'))

model.add(Convolution2D(12, 1,1))
model.add(Activation('relu'))

##MaxPooling Layer - To Downsample ## (b,24,24,16)
model.add(MaxPooling2D(pool_size=(2,2)))

## Convolution Block 3 ## (b,12,12,16)
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))

##Shape :- (b, 10, 10, 8)

##Flatten
model.add(Flatten())

##Shape :- (b,800)

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train = X[:split].reshape((-1,28,28,1))
print X_train.shape
X_test = X[split:].reshape((-1, 28, 28,1))
print X_test.shape


split = int(0.8*X.shape[0])
hist = model.fit(X_train, Y[:split], nb_epoch=20, batch_size=16,validation_data=(X_test, Y[split:]), verbose=2)