# -*- coding: utf-8 -*-
# @Time : 2020/4/2 下午5:42
# @Author : LuoLu
# @FileName: binary_image.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import matplotlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL.Image import Image


path = "/home/luolu/PycharmProjects/ParticleDetection/data/image/particle_ct.png"
original = cv.imread(path)
height, width, channels = original.shape
src = cv.GaussianBlur(original, (1, 1), 0)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# cv.THRESH_BINARY | cv.THRESH_OTSU        "ct.png"
ret, binary_ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TOZERO)

# 使用开运算去掉外部的噪声
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
# binary = cv.morphologyEx(binary_, cv.MORPH_OPEN, kernel)
cv.imshow('binary', binary_)

cv.imwrite("/home/luolu/PycharmProjects/ParticleDetection/data/image/binary_ct.png", binary_)

cv.waitKey(0)
cv.destroyAllWindows()