# -*- coding: utf-8 -*-
# @Time : 2020/4/8 上午11:08
# @Author : LuoLu
# @FileName: skeleton_img.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

# Import the necessary libraries
import cv2
import numpy as np

# Read the image as a grayscale image
img = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/test.png', 0)

# img = cv2.GaussianBlur(img, (3, 3), 0)

# Threshold the image
# ret, img = cv2.threshold(img, 40, 255, 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

# Step 1: Create an empty skeleton
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Repeat steps 2-4
while True:
    # Step 2: Open the image
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    # Step 3: Substract open from the original image
    temp = cv2.subtract(img, open)
    # Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img) == 0:
        break

# Displaying the final skeleton
cv2.namedWindow("Skeleton", flags=2)
cv2.imshow("Skeleton", skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
