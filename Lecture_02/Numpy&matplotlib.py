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


ones = np.ones((4,2), dtype=np.uint8)
print np.unique(ones)

a = np.array(range(1,100))
#plt.figure(0)
plt.plot(a,'r>')
plt.show()
#plt.figure(1)
plt.plot(a*2,a,'g+')
#plt.plot(a,'g+')
#plt.show()

arr = np.random.random((1000,2))
arr1 = np.random.random((1000000,2))
plt.scatter(arr[:,0], arr[:,1],color='black')
#plt.scatter(arr1[:,0], arr1[:,1], color='green')
plt.show()



