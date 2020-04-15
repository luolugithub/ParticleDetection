# -*- coding: utf-8 -*-
# @Time : 2020/4/5 下午3:55
# @Author : LuoLu
# @FileName: watershed_segmentation.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from PIL import Image

img = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/Bitwiseand_test.png', 0)

# '''Adaptive thersholding
#    calculates thresholds in regions of size block_size surrounding each pixel
#    to handle the non-uniform background'''
# block_size = 31
# adaptive_thresh = threshold_local(img, block_size)  # , offset=10)
# binary_adaptive = img > adaptive_thresh

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary_ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
# cv2.imwrite('/home/luolu/PycharmProjects/ParticleDetection/data/image/binary_37.png', Image.fromarray(
# binary_adaptive))


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(binary_, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area
# cv2.imshow('unknown', unknown)

# Perform the distance transform algorithm
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L1, 5)

# Calculate Euclidean distance
distance = ndi.distance_transform_edt(img)

# Find local maxima of the distance map
local_maxi = peak_local_max(dist_transform, labels=img, footprint=np.ones((51, 51)), indices=False)
# Label the maxima
markers = ndi.label(local_maxi)[0]

''' Watershed algorithm
    The option watershed_line=True leave a one-pixel wide line 
    with label 0 separating the regions obtained by the watershed algorithm '''
labels = watershed(-dist_transform, markers, watershed_line=True, connectivity=8)

# Plot the result
print(type(labels))
print(labels.shape)
plt.imshow(img, cmap='gray')
plt.imshow(labels == 0, alpha=0.3, cmap='Reds')
plt.show()

img = Image.fromarray(labels == 0)
img.save('/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/wline_test_bw.png')
