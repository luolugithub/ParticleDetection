# -*- coding: utf-8 -*-
# @Time : 2020/4/1 上午10:37
# @Author : LuoLu
# @FileName: intersections_contours.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import numpy as np
import cv2
import matplotlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL.Image import Image


def getJunctions(src):
    # the hit-and-miss kernels to locate 3-points junctions to be used in each directions
    # NOTE: float type is needed due to limitation/bug in warpAffine with signed char
    k1 = np.asarray([
        0, 1, 0,
        0, 1, 0,
        1, 0, 1], dtype=float).reshape((3, 3))
    k2 = np.asarray([
        1, 0, 0,
        0, 1, 0,
        1, 0, 1], dtype=float).reshape((3, 3))
    k3 = np.asarray([
        0, -1, 1,
        1, 1, -1,
        0, 1, 0], dtype=float).reshape((3, 3))

    # Some useful declarations
    tmp = np.zeros_like(src)
    ksize = k1.shape
    center = (ksize[1] / 2, ksize[0] / 2)  # INVERTIRE 0 E 1??????
    # 90 degrees rotation matrix
    rotMat = cv2.getRotationMatrix2D(center, 90, 1)
    # dst accumulates all matches
    dst = np.zeros(src.shape, dtype=np.uint8)

    # Do hit & miss for all possible directions (0,90,180,270)
    for i in range(4):
        tmp = cv2.morphologyEx(src, cv2.MORPH_HITMISS, k1.astype(np.int8), tmp, (-1, -1), 1, cv2.BORDER_CONSTANT, 0)
        dst = cv2.add(dst, tmp)
        tmp = cv2.morphologyEx(src, cv2.MORPH_HITMISS, k2.astype(np.int8), tmp, (-1, -1), 1, cv2.BORDER_CONSTANT, 0)
        dst = cv2.add(dst, tmp)
        tmp = cv2.morphologyEx(src, cv2.MORPH_HITMISS, k3.astype(np.int8), tmp, (-1, -1), 1, cv2.BORDER_CONSTANT, 0)
        dst = cv2.add(dst, tmp)
        # Rotate the kernels (90deg)
        k1 = cv2.warpAffine(k1, rotMat, ksize)
        k2 = cv2.warpAffine(k2, rotMat, ksize)
        k3 = cv2.warpAffine(k3, rotMat, ksize)

    return dst


# This is your sample image (objects are black)
# src = np.asarray([0, 1, 1, 1, 1, 1, 0, 0,
#                   1, 0, 1, 1, 1, 0, 1, 1,
#                   1, 1, 0, 0, 0, 1, 1, 1,
#                   1, 1, 1, 0, 0, 0, 1, 1,
#                   1, 0, 0, 1, 1, 1, 0, 1,
#                   0, 1, 1, 1, 1, 1, 0, 0,
#                   0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8).reshape((7, 8))


path = "/home/luolu/PycharmProjects/ParticleDetection/data/image/test.png"
original = cv.imread(path)
height, width, channels = original.shape
src = cv.GaussianBlur(original, (5, 5), 0)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary_ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

# 使用开运算去掉外部的噪声
kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
binary = cv.morphologyEx(binary_, cv.MORPH_OPEN, kernel)
print(type(binary))
src = np.asarray(binary, dtype=np.uint8).reshape(242, 244)
print(binary.shape)


# src *= 255
# Morphology logic is: white objects on black foreground
# src = 255 - src

# Get junctions
junctionsScore = getJunctions(src)

# Draw markers where junction score is non zero
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
# find the list of location of non-zero pixels
junctionsPoint = cv2.findNonZero(junctionsScore)

for pt in junctionsPoint:
    pt = pt[0]
    dst[pt[1], pt[0], :] = [0, 0, junctionsScore[pt[1], pt[0]]]

# show the result
winDst = "Dst"
winSrc = "Src"
winJunc = "Junctions"
cv2.namedWindow(winSrc, flags=2)
cv2.namedWindow(winJunc, flags=2)
cv2.namedWindow(winDst, flags=2)
scale = 24
# cv2.resizeWindow(winSrc, scale * src.shape[1], scale * src.shape[0])
# cv2.resizeWindow(winJunc, scale * src.shape[1], scale * src.shape[0])
# cv2.resizeWindow(winDst, scale * src.shape[1], scale * src.shape[0])
cv2.imshow(winSrc, original)
cv2.imshow(winJunc, junctionsScore)
cv2.imshow(winDst, dst)
cv2.waitKey()
