import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline


im = cv2.imread('/home/Stonedcoder/Perceptron_Summe/class_11/img.jpg')
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')
plt.show()