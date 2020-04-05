# -*- coding: utf-8 -*-
# @Time : 2020/4/2 下午3:10
# @Author : LuoLu
# @FileName: watershed_test.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import cv2
import numpy as np
from Watershed import Watershed
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2
import numpy as np

image = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/thresh_pills_02.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imwrite("/home/luolu/PycharmProjects/ParticleDetection/data/image/thresh_pills_02.png", thresh)
# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area
cv2.imshow('unknown', unknown)

# Perform the distance transform algorithm
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L1, 5)
# Normalize the distance image for range = {0.0, 1.0}
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

# Finding sure foreground area
ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers_copy = markers.copy()
markers_copy[markers == 0] = 255  # 灰色表示背景
markers_copy[markers == 1] = 0  # 黑色表示背景
markers_copy[markers > 1] = 255  # 白色表示前景

markers_copy = np.uint8(markers_copy)

# 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
markers = cv2.watershed(image, markers)
image[markers == -1] = [0, 0, 255]  # 将边界标记为红色
img_markers = Image.fromarray(markers, 'RGB')
print(image.shape)
print(type(markers))
print(markers.shape)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_thresh = cv2.threshold(image_gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
subtract = cv2.add(np.asarray(image_thresh), np.asarray(thresh))
unknown_opening = cv2.bitwise_and(np.asarray(unknown), np.asarray(thresh))
unknown_opening_thresh = cv2.bitwise_xor(np.asarray(image_thresh), np.asarray(thresh))
# subtract = subtract.convert('RGBA')
# subtract = Image.fromarray(subtract)

cv2.imshow('image', image)
cv2.imshow('thresh', thresh)
# cv2.imshow('markers', img_markers)
cv2.imshow('opening', opening)
cv2.imshow('markers_copy', markers_copy)
cv2.imshow('subtract', subtract)
cv2.imshow('unknown_opening', unknown_opening)
cv2.imshow('unknown_opening_thresh', unknown_opening_thresh)
cv2.waitKey()
