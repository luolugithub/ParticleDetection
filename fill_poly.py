# -*- coding: utf-8 -*-
# @Time : 2020/3/23 下午4:17
# @Author : LuoLu
# @FileName: fill_poly.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2

# Load in image, convert to grayscale, and threshold
image = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/background_line_pills.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Close contour
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1000, 1000))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find outer contour and fill with white
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(close, cnts, [255, 255, 255])

cv2.imshow('thresh', thresh)
cv2.imshow('close', close)
cv2.waitKey()
