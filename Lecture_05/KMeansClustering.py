import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Define a dataset
mean_01 = np.array([1.0, 0.5])
cov_01 = np.array([[1.0,0.1],[0.1,1.2]])

mean_02 = np.array([5.3,6.5])
cov_02 = np.array([[1.0,0.1],[0.1,1.2]])

dist_01 = np.random.multivariate_normal(mean_01, cov_01, 500)
dist_02 = np.random.multivariate_normal(mean_02, cov_02, 500)

print dist_01.shape, dist_02.shape


plt.scatter(dist_01[:,0],dist_01[:,1], color='red')
plt.scatter(dist_02[:,0],dist_02[:,1], color='blue')

plt.show()


## Input data 
## (dist_01.shape[0]+dist_02.shape[0], #num_features)
data = np.concatenate((dist_01, dist_02))
print data.shape

print data.min(), data.max()