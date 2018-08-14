import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential

ds = pd.read_csv('../../mnist_train.csv')
ds.tail()



data = ds.values[6000:17000]
print data.shape


#X = data[:,1:]
X = data[:,1:]/255.0
y = data[:, 0]
print X.shape
print y.shape

print np.unique(y)
nb_classes = len(np.unique(y))


Y = np_utils.to_categorical(y)
for ix in range(10):
    print y[ix], Y[ix]
print Y.shape



split = int(0.8*data.shape[0])
X_train = X[:split]
X_test = X[split:]

y_train = Y[:split]
y_test = Y[split:]

print X_train.shape, y_train.shape



model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()




model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, nb_epoch=20, batch_size=50, validation_data=(X_test, y_test), verbose=2)


for layer in model.layers:
    print layer
    if len(layer.get_weights())>0:
        print layer.get_weights()[0].shape
        print layer.get_weights()[1].shape
    print layer.get_config()['name']
    print "-----------------"



    