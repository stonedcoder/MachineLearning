import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.preprocessing import LabelEncoder



ds = pd.read_csv('../../Machine-Learning-with-R-datasets/mushrooms.csv')
ds.head()



le = LabelEncoder()
df = ds.apply(le.fit_transform) 
df.head()



split = int(0.9*df.values.shape[0])
train = df.loc[:split]
print type(train)
print type(df.loc)
test = df.loc[split:]
test = test.reset_index().drop('index', axis=1)




print type(df['type'])
#print df['type']
count = 0
print len(df['type'])
x = np.unique(np.array(df['type']), return_counts=True)
print x
for ix in x[1]:
    print float(ix)/x[1].sum()



    def prior_probability(data, label_column, label_value):
    n_rows = data.loc[data[label_column]==label_value]
    return float(n_rows.shape[0])/data.shape[0]

    def conditional_probability(data, feature_column, feature_value, label_column, label_value):
    satisfiable_constraint = data.loc[(data[label_column]==label_value) & (data[feature_column]==feature_value)]
    n_rows = data.loc[data[label_column]==label_value]
    return float(satisfiable_constraint.shape[0])/n_rows.shape[0]
    print conditional_probability(train, 'cap_shape', 5, 'type', 1)



py = prior_probability(train, 'type', 1)
print py
cp = conditional_probability(train,'cap_shape', 5, 'type', 1)
print cp    



preds = []
CLASSES = [0, 1]
for ix in range(test.shape[0]):
    #print ix
    val = test.loc[ix][1:]
    feature = dict(val)
    likelihoods = []
    for cx in CLASSES:
        likelihood = 1.0
        for fx in feature:
            cp = conditional_probability(train, fx, feature[fx],'type',cx)
            likelihood *= cp
        
        prior = prior_probability(train, 'type', cx)
        
        likelihoods.append(prior*likelihood)
    
    preds.append(np.array(posterior).argmax())



(preds == test['type']).sum()/ float(test['type'].shape[0])
