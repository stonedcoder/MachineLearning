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

