import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline

##reading titanic .
ds = pd.read_csv('./titanic.csv')
print ds.columns
print len(ds.columns)
ds.head()


cols_to_remove = [
    'PassengerId',
    'Name',
    'Cabin',
    'Embarked',
    'Ticket'
]
df = ds.drop(cols_to_remove, axis=1)
print len(df.columns)

df.head()

# sex mapping .
sex_mapping = {
    'male': 0,
    'female': 1,
}
df.Sex = df.Sex.map(sex_mapping)


data = df.values
print data.shape

df = df.dropna(axis=0)

data = df.values
print data.shape