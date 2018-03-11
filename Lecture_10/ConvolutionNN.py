import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline


im = cv2.imread('/home/Stonedcoder/Perceptron_Summe/class_11/img.jpg')
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')
plt.show()

K = np.array([[-0,-1, 0],[-1, 4, -1], [0,-1,0]])
print K
print K.shape

out_r = img.shape[0] - K.shape[0] + 1
out_c = img.shape[1] - K.shape[1] + 1
X = np.zeros((out_r, out_c))
print img.shape
print X.shape

