import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

my_list = [1,2,3,4]
print type(my_list)
my_listarr = np.array(my_list)
print my_listarr
print type(my_listarr)
print my_listarr.shape

arr = np.zeros((4,2), dtype=np.uint8)
print arr.shape
print arr.dtype

my_arr = np.zeros((4,2))
print my_arr.shape
my_arr[:,0] +=1.0
my_arr[:,1] +=2.0
print np.unique(my_arr, return_counts=True)
# print my_arr