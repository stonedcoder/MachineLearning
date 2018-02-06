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

def entropy(dataset, col=0):
    p = []
    
    p_survive = dataset[:, col].mean()
    p.append(p_survive)
    
    p.append(1-p[0])
    
    ent = 0.0
    for px in p:
        ent += (-1.0*px*np.log2(px))
    
    return ent


def information_gain(dataset,subset):
    return entropy(dataset) - entropy(subset)

def information_gain2(dataset, left_subset, right_subset):
    
    left = left_subset
    right = right_subset
    
    h_data = entropy(dataset)
    if left.shape[0]>0:
        left_gain = h_data - entropy(left)
    else:
        left_gain = -100000
        
    if right.shape[0]>0:
        right_gain = h_data - entropy(right)
    else:
        right_gain = -100000
    
    total_gain = right_gain + left_gain
    return total_gain



INFINITY = -10000000
class DT:
    def __init__(self, depth, max_depth):
        self.right = None
        self.left = None
        self.col_id = None
        self.value = None
        self.depth = depth
        self.max_depth = max_depth
    
    def select_attr_And_run(self, dataset):
        
        ## Exit Condition
        if self.depth>= self.max_depth:
            return 
        
        n_cols = 6
        start_index = 1
        check_index = 0
        
        all_gains = []
        
        for cx in range(start_index, start_index+n_cols):
            split_val = dataset[:, cx].mean()
            
            right, left = data_split(dataset, cx, split_val)
            
            if left.shape[0] > 0:
                left_gain = information_gain(dataset, left)
            else:
                left_gain = INFINITY
            
            if right.shape[0] > 0:
                right_gain = information_gain(dataset, right)
            else:
                right_gain = INFINITY
            
            comb_gain = right_gain + left_gain
            all_gains.append(comb_gain)
        
        all_gains = np.array(all_gains)
        self.col_id = np.argmax(all_gains) + start_index
        self.value = dataset[:, self.col_id].mean()
        
        data_right, data_left = data_split(dataset, self.col_id, self.value)
        
        if data_right.shape[0] > 0:
            self.right = DT(self.depth+1,self.max_depth)
            self.right.select_attr_And_run(data_right)
        
        if data_left.shape[0] > 0:
            self.left = DT(self.depth+1, self.max_depth)
            self.left.select_attr_And_run(data_left)
        
        return 
    
    def predict(example):
        ## example of shape :- (1, #features)
        ## Here -> (1,6)
        pass    


