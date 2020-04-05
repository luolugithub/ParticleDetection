# -*- coding: utf-8 -*-
# @Time : 2020/3/25 上午10:11
# @Author : LuoLu
# @FileName: objects_counting.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

path = "data/zhutibp_cui0326/1.jpg"
original = cv.imread(path)
# Convert image in grayscale
gray_im = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
plt.figure(1, figsize=(8, 8))
plt.tight_layout()
plt.subplot(221)
plt.title('Grayscale image')
plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

# Contrast adjusting with gamma correction y = 1.2

gray_correct = np.array(255 * (gray_im / 255) ** 1.2, dtype='uint8')
plt.subplot(222)
plt.title('Gamma Correction y= 1.2')
plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
# Contrast adjusting with histogramm equalization
gray_equ = cv.equalizeHist(gray_im)
plt.tight_layout()
plt.subplot(223)
plt.title('Histogram equilization')
plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
# plt.show()

# Local adaptative threshold

thresh = cv.adaptiveThreshold(gray_correct, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 0)
thresh = cv.bitwise_not(thresh)
plt.subplot(224)
plt.title('Local adapatative Threshold')
plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)
# plt.show()

# Dilatation et erosion
kernel = np.ones((0, 0), np.uint8)
img_dilation = cv.dilate(thresh, kernel, iterations=1)
img_erode = cv.erode(img_dilation, kernel, iterations=1)
# clean all noise after dilatation and erosion
img_erode = cv.medianBlur(img_erode, 1)
plt.figure(2, figsize=(8, 8))
plt.subplot(221)
plt.title('Dilatation + erosion')
plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)
# plt.show()
# Labeling

ret, labels = cv.connectedComponents(img_erode, labels=0)
connectivity = 8  # You need to choose 4 or 8 for connectivity type
output = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
# print('ret:', ret)
print('labels:', labels)
label_hue = np.uint8(255 * labels / np.max(labels))
blank_ch = 255 * np.zeros_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
labeled_img[label_hue == 255] = 255

plt.subplot(222)
plt.title('Objects counted:' + str(ret - 1))
plt.imshow(labeled_img)
print('objects number is:', ret - 1)
plt.show()
