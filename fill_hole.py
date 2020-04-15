# -*- coding: utf-8 -*-
# @Time : 2020/4/13 下午3:59
# @Author : LuoLu
# @FileName: fill_hole.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2
import numpy as np

# read image, ensure binary
img = cv2.imread('data/yashi_qscan/mask/lb_mask.png', 0)
img[img != 0] = 255

# flood fill background to find inner holes
holes = img.copy()
cv2.floodFill(holes, None, (0, 0), 255)

# invert holes mask, bitwise or with img fill in holes
holes = cv2.bitwise_not(holes)
filled_holes = cv2.bitwise_or(img, holes)
cv2.imshow('filled_holes', filled_holes)
cv2.waitKey()
