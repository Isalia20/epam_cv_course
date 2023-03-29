import cv2 as cv
import numpy as np

img = cv.imread("/Users/iraklisalia/Desktop/Screenshot 2023-03-15 at 00.18.55.png")

px = img[100, 100]
blue = img[100, 100, 0]
img[100, 100]= [ 255, 255, 255]

# Accessing red value
img.item(10, 10, 2)
# Modifying red value
img.itemset((10, 10, 2), 100)
img.item(10, 10, 2)

img.size

ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

# Splitting and merging image channels
b, g, r = cv.split(img)
# or by using numpy indexing
b = img[:, :, 0]

# Set red colors to 0
img[:, :, 2] = 0
cv.useOptimized()


flags =[i for i in dir(cv) if i.startswith('COLOR_')]

flags



import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


green = np.uint8([[[0,255,0 ]]])

hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)

print(hsv_green)