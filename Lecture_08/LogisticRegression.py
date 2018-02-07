import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

mean_01 = np.array([1.0, 0.5])
mean_02 = np.array([4.0, 5.2])

cov_01 = np.array([[1.0,0.1], [0.1, 1.0]])
cov_02 = np.array([[1.0,0.1], [0.1, 1.2]])

dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)

print dist_01.shape, dist_02.shape


rows = dist_01.shape[0] + dist_02.shape[0]
cols = dist_01.shape[1] + 1

data = np.zeros((rows, cols))
print data.shape

data[:dist_01.shape[0], :2] = dist_01
data[dist_01.shape[0]:, :2] = dist_02
data[dist_01.shape[0]:, -1] += 1.0

print data.shape

np.random.shuffle(data)

print data[:10]


split = int(0.65*data.shape[0])

X_train = data[:split, :2]
X_test = data[split:, :2]

y_train = data[:split, -1]
y_test = data[split:, -1]

print X_train.shape, y_train.shape
print X_test.shape, y_test.shape