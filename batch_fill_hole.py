# -*- coding: utf-8 -*-
# @Time : 2020/4/14 上午10:07
# @Author : LuoLu
# @FileName: batch_fill_hole.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import glob
import os

import cv2 as cv
import numpy as np

root_path = '/home/luolu/PycharmProjects/ParticleDetection/'

if __name__ == '__main__':
    base_name = ''
    counter = 0
    for filename in glob.glob('data/yashi_qscan/mask/*.png'):
        img = cv.imread(filename, 0)
        # height, width, channels = img.shape
        print(filename)
        base_name = os.path.basename(filename)
        save_name = base_name.split('_')[0]

        # fill hole
        # read image, ensure binary
        img[img != 0] = 255

        # flood fill background to find inner holes
        holes = img.copy()
        cv.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv.bitwise_not(holes)
        filled_holes = cv.bitwise_or(img, holes)

        cv.imwrite(root_path + "data/yashi_qscan/" + save_name + '.png', filled_holes)
        counter = counter + 1

    print('counter: ', counter)
