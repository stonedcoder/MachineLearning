import matplotlib.pyplot as plt
import  numpy as np
%matplotlib inline

mean_01 = [0.5, 1.0]
mean_02 = [5.0, 6.0]

cov_01 = [[0.5, 0.1], [0.1, 0.5]]
cov_02 = [[1.0, 0.1], [0.1, 1.0]]

dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)


plt.scatter(dist_01[:, 0], dist_01[:, 1], color='red')
plt.scatter(dist_02[:, 0], dist_02[:, 1], color='green')
plt.show()


rows = dist_01.shape[0] + dist_02.shape[0]
cols = dist_01.shape[1] + 1

data = np.zeros((rows, cols))
data[:dist_01.shape[0],:2] = dist_01
data[dist_01.shape[0]:, :2] = dist_02
data[dist_01.shape[0]:, -1] = 1.0
print data.shape

np.random.shuffle(data)
test_data = data[:10]


def distance_euclid(p1, p2):
    ## Euclidian
    d = np.sqrt(((p1-p2)**2).sum())
    return d


def distance(p1, p2):
   
 #Manhattan Distance    d = (abs(p1-p2)).sum()
    return d

distance(np.array([1.0,1.0]), np.array([3.0, 3.0]))


def KNN(X_train, Y_train, xtest, k=5):
    vals = []
    for ix in range(X_train.shape[0]):
        d = distance_euclid(X_train[ix], xtest)
        vals.append([d,Y_train[ix]])
    
    sorted_vals = sorted(vals, key=lambda mn:mn[0])
    neighbours = np.array(sorted_vals)[:k,-1]
    freq = np.unique(neighbours, return_counts=True)
    
    my_ans = freq[0][freq[1].argmax()]
    return my_ans

### Dataset for input
X_train = test_data[:,:2]
Y_train = test_data[:, -1]
x_test = np.array([2.9, 2.9])
k = 3
ans = KNN(X_train, Y_train, x_test, 3)

print ans


## Test and Train Split
split = int(0.60*data.shape[0])

train_x = data[:split,:2]
train_y = data[:split, -1]

print np.unique(train_y, return_counts=True)

test_x = data[split:, :2]
test_y = data[split:, -1]

print train_x.shape
print train_y.shape 


def get_acc(kx,x_train,x_test,y_train, y_test):
    preds = []
    for ix in range(x_test.shape[0]):
        label = KNN(x_train, y_train, x_test[ix], k=kx)
        preds.append(label)
    preds = np.array(preds)
    
    return 100*float((preds==y_test).sum())/y_test.shape[0] 

for kx in range(3, 9, 2):
    print kx, " | ", get_acc(kx,train_x,test_x, train_y, test_y)


import pandas as pd
### Fashion dataset for MNIST
ds = pd.read_csv('./fashion-mnist_train.csv')
ds.tail(n=3)
fashion_data = ds.values[:3000]
print fashion_data.shape



get_acc(3,fashion_train_x, fashion_test_x, fashion_train_y, fashion_test_y)
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(fashion_train_x, fashion_train_y)
100*neigh.score(fashion_test_x, fashion_test_y)

plt.imshow(fashion_train_x[1].reshape(28,28), cmap='gray')
plt.show()
print fashion_train_y[1]
