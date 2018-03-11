import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.utils import np_utils
import pandas as pd

ds = pd.read_csv('../../mnist_train.csv')
data = ds.values[100:4600,1:]/255.0
print data.shape


inp = Input(shape=(784,))
h1 = Dense(100)
a1 = Activation('sigmoid')
y = Dense(784)
ya = Activation('sigmoid')

yout =  ya(y(a1(h1(inp))))
model = Model(input=[inp], output=[yout])
model.summary()


model.compile(optimizer='adam', loss='MSE', metrics=['accuracy'])

split = int(0.8*data.shape[0])

model.fit(data[:split], data[:split],nb_epoch=60, batch_size=30,verbose=2, validation_data=(data[split:], data[split:]))



##Encoder 
enc_out = a1(h1(inp))
encoder = Model(input=[inp], output=[enc_out])

##Decoder
dec_inp = Input(shape=(100,))
dec_out = ya(y(a1(dec_inp)))
decoder = Model(input=[dec_inp], output=[dec_out])