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


