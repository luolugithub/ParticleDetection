# -*- coding: utf-8 -*-
# @Time : 2020/3/5 上午11:18
# @Author : LuoLu
# @FileName: canny_Edge_Detection.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com


import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt


img = cv.imread('data/bp_src3/5.jpg', 0)
edges = cv.Canny(img, 720, 576, edges=200, apertureSize=7)
edges = Image.fromarray(edges)
edges = edges.convert('RGBA')
data = np.array(edges)   # "data" is a height x width x 4 numpy array
red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

# Replace white with red... (leaves alpha values alone...)
# white_areas = (red == 255) & (blue == 255) & (green == 255)
# data[..., :-1][white_areas.T] = (0, 136, 126)     # Transpose back needed

im2 = Image.fromarray(data)
im2.save("data/temp/edge_ct.png")
im2.show()

