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

#Data shapes . 
data = df.values
print data.shape

df = df.dropna(axis=0)

data = df.values
print data.shape


## printing test and train set images
X = data[:,1:] ## input
Y = data[:, 0] ## labels

print X.shape
print Y.shape

split = int(0.8*X.shape[0])

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

print "-----------"

print X_train.shape
print X_test.shape

print Y_train.shape
print Y_test.shape


### Getting started with decision tree classifier 
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
print dt.score(X_test, Y_test)

## Random forest
rf = RandomForestClassifier(n_estimators=145)
rf.fit(X_train, Y_train)
print rf.score(X_test, Y_test)


### decision tree
print data.shape

def data_split(dataset, col, value):
    data_right = []
    data_left = []
    
    for ix in range(dataset.shape[0]):
        
        if dataset[ix, col] > value:
            data_right.append(data[ix,:])
        else:
            data_left.append(data[ix, :])
    
    return np.array(data_right), np.array(data_left)




