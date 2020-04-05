# -*- coding: utf-8 -*-
# @Time : 2020/4/2 下午7:33
# @Author : LuoLu
# @FileName: segment_watershed.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import cv2 as cv
import numpy as np

src = cv.imread("/home/luolu/PycharmProjects/ParticleDetection/data/image/distance_pills_02.png")
cv.imshow("input", src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("binary", binary)

# 形态学操作
se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
open_img = cv.morphologyEx(binary, cv.MORPH_OPEN, se, iterations=2)

# sure background area
sure_bg = cv.dilate(open_img, se, iterations=3)
cv.imshow("sure_bg", sure_bg)

# 距离变换
dist_transform = cv.distanceTransform(open_img, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
cv.imshow("distance transform", dist_transform)
cv.imshow("sure_fg", sure_fg)

print(type(dist_transform))
print(type(binary))
dist_binary = cv.subtract(dist_transform, binary)
cv.imshow("dist_binary", dist_binary)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow("unknown", unknown)

# 连通组件标记 - 发现markers
ret, markers = cv.connectedComponents(sure_bg)
markers = markers + 1

# 设定边缘待分割区域
markers[unknown == 255] = 0

_, dist = cv.threshold(dist_transform, 0.4, 1.0, cv.THRESH_BINARY)

# Dilate a bit the dist image
kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
cv.imshow("dist", dist)

# 分水岭分割
markers = cv.watershed(src, markers)
# markers = np.zeros(src.shape, dtype=np.uint8)
# line = cv.watershed(src, dist_transform)
# src[markers == -1] = [0, 0, 255]
src[markers == -1] = [0, 0, 0]
# cv.imshow("line", line)
cv.imshow("result", src)
# cv.imshow("markers", markers)

# print(dist_transform.shape)
# print(binary.shape)
# gray_src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# ret, binary_src = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# result_binary = cv.add(binary_src, np.asarray(binary))
# cv.imshow("result_binary", result_binary)
print(type(dist_transform))
print(type(binary))
# dist_transform_binary = cv.add(np.asarray(dist_transform), np.asarray(binary))
# cv.imshow("dist_transform_binary", dist_transform_binary)


cv.waitKey(0)
cv.destroyAllWindows()
