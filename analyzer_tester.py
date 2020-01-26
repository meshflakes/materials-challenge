import numpy as np
import cv2


# Read the image and create a blank mask
img = cv2.imread('ag-sn.png')
h,w = img.shape[:2]
mask = np.zeros((h,w), np.uint8)

# Transform to gray colorspace and threshold the image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Perform opening on the thresholded image (erosion followed by dilation)
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Search for contours and select the biggest one and draw it on mask
contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)
cv2.drawContours(mask, [cnt], 0, 255, -1)

# Perform a bitwise operation
res = cv2.bitwise_and(img, img, mask=mask)

# Threshold the image again
gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find all non white pixels
non_zero = cv2.findNonZero(thresh)

# Transform all other pixels in non_white to white
for i in range(0, len(non_zero)):
    first_x = non_zero[i][0][0]
    first_y = non_zero[i][0][1]
    first = res[first_y, first_x]
    res[first_y, first_x] = 255

# Display the image
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
