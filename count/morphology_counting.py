# -*- coding: utf-8 -*-
# @Time : 2020/4/7 下午4:28
# @Author : LuoLu
# @FileName: morphology_counting.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
from scipy.ndimage import measurements, morphology
from PIL import Image
from numpy import *
from scipy.ndimage import filters
import cv2 as cv

path = "/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/edge/edge_cl.png"
original = cv.imread(path)
src = cv.GaussianBlur(original, (3, 3), 0)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret, binary_ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)

# 使用开运算去掉外部的噪声
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
binary = cv.morphologyEx(binary_, cv.MORPH_OPEN, kernel)
# load image and threshold to make sure it is binary
im = array(Image.open(path))
im = 1 * (im < 200)

labels, nbr_objects = measurements.label(im)
print("Number of objects1:", nbr_objects)

# morphology - opening to separate objects better
im_open = morphology.binary_opening(binary, ones((1, 1)), iterations=2)
im_open = Image.fromarray(im_open)
Image._show(im_open)
# print(type(im_open))
labels_open, nbr_objects_open = measurements.label(im_open)
labels_open = Image.fromarray(labels_open == 0)
Image._show(labels_open)
# print(type(labels_open))
# print(type(nbr_objects_open))
print("morphology Number of objects:", nbr_objects_open)

