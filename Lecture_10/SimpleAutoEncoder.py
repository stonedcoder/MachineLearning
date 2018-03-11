import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.utils import np_utils
import pandas as pd

ds = pd.read_csv('../../mnist_train.csv')
data = ds.values[100:4600,1:]/255.0
print data.shape