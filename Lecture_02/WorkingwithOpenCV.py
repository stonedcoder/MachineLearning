import cv2
import inline as inline
import matplotlib as matplotlib
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


# 1.3: Applying Canny Edge Detection
edged = cv2.Canny(blurred, 0, 50)
plt.figure(3, figsize=(7,7))
plt.imshow(edged, cmap='gray')

plt.show()


# Step 2: Finding largest contour in Edged Image

# 2.1: Find Contours
(_, contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NO)

# 2.2 Sort contours by area in decreasing order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Plotting a bounding rectangle around largest contour for representation purposes
x,y,w,h = cv2.boundingRect(contours[0])
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
plt.figure(4, figsize=(7,7))
plt.imshow(image, cmap='gray')
plt.show()


# 2.3 Getting largest approximate contour with 4 vertices
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break

print 'Largest approximate Contour is: '
print target


# Plotting the largest contour for representation 
cv2.drawContours(image, [target], -1, (255, 0, 0), 2)
plt.figure(5, figsize=(7,7))
plt.imshow(image, cmap='gray')
plt.show()



# Step 3: Mapping (Transforming) target points to 800x800 quadrilateral

approx = rectify(target)
print '\nLargest approximate Contour after rectification is: '
print approx