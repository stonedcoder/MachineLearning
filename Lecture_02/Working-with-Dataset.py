import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

ds = pd.read_csv('./movie_metadata.csv')
print type(ds)
print len(ds.columns)

data = ds.values
print type(data)
print data.shape
#ds.head(n=5)
ds.tail(n=3)


#for ix in ds.columns:
   #print ix
print type(ds.columns)

titles = ds.get('movie_title')
print len(titles)
print type(titles)

