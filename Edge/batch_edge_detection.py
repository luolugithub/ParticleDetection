# -*- coding: utf-8 -*-
# @Time : 2020/3/19 下午6:24
# @Author : LuoLu
# @FileName: batch_edge_detection.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import glob
import os

import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt

root_path = '/home/luolu/PycharmProjects/color_detection'

if __name__ == '__main__':
    base_name = ''
    for filename in glob.glob(root_path + '/data/result_qaak/lb/*.png'):
        img = cv.imread(filename)
        base_name = os.path.basename(filename)
        height, width, channels = img.shape
        print(height, width, channels)
        edges = cv.Canny(img, height, width, edges=200, apertureSize=7)
        print(type(edges))
        edges = Image.fromarray(edges)
        edges = edges.convert('RGBA')
        data = np.array(edges)  # "data" is a height x width x 4 numpy array
        red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

        # Replace white with red... (leaves alpha values alone...)
        white_areas = (red == 255) & (blue == 255) & (green == 255)
        # light Blue 0, 255, 255
        # caolv 0, 128, 128
        # fen 255, 192, 192
        data[..., :-1][white_areas.T] = (0, 255, 255)     # Transpose back needed

        im2 = Image.fromarray(data)
        im2.save(root_path + "/data/result_qaak/edge_lb/" + base_name)