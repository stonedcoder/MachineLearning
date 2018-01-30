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



# Add image here. Can also use laptop's webcam if the resolution is good enough to capture readable document content
image = cv2.imread('./sample-1.jpg')
print image.shape

# Resize image so it can be processed. Choose optimal dimensions such that important content is not lost
image = cv2.resize(image, (1500, 880))
orig = image.copy()

plt.figure(0)
plt.imshow(image)
plt.show()

print image.shape


# Step 1: Edge Detection

# 1.1: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(1, figsize=(7,7))
plt.imshow(gray, cmap='gray')

# 1.2: Blurring for Smoothness: 
# #Gaussian Blur,   

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
plt.figure(2, figsize=(7,7))
plt.imshow(blurred, cmap='gray')

# Median Blur
blurred = cv2.medianBlur(gray, 5)
plt.figure(2, figsize=(7,7))
plt.imshow(blurred, cmap='gray')
