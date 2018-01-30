import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def rectify(h):
    # print h
    h = h.reshape((4,2))
    # print h
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    # print add
    
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew