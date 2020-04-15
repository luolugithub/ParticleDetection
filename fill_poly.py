# -*- coding: utf-8 -*-
# @Time : 2020/3/23 下午4:17
# @Author : LuoLu
# @FileName: fill_poly.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2
from skimage import morphology
import numpy as np

# Load in image, convert to grayscale, and threshold
image = cv2.imread('data/yashi_qscan/mask/fen_mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Close contour
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
# open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


# Find outer contour and fill with white
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(close, cnts, [255, 255, 255])
print(type(close))
# binary = cv2.threshold(close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cleaned = morphology.remove_small_objects(binary, min_size=64, connectivity=8)

# remove_small_objects

# find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(close, connectivity=8)
# connectedComponentswithStats yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 250

# your answer image
img2 = np.zeros((output.shape))
# for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

cv2.imshow('thresh', thresh)
cv2.imshow('close', close)
cv2.imshow('cleaned', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
