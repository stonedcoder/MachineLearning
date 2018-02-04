import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline


ds = pd.read_csv('./titanic.csv')
print ds.columns
print len(ds.columns)
ds.head()