# -*- coding: utf-8 -*-
# @Time : 2020/3/12 下午1:57
# @Author : LuoLu
# @FileName: batch_composite_img.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com


from PIL import Image
import glob
import os

import cv2 as cv
import numpy as np
from PIL import Image
import numpy as np

if __name__ == '__main__':
    background = []
    foreground1 = []
    foreground2 = []
    foreground3 = []
    base_name = ''
    counter = 0
    for filename in sorted(glob.glob('/home/luolu/PycharmProjects/color_detection/data/result_qaak/transparent/*.png')):
        img = cv.imread(filename)
        height, width, channels = img.shape
        # print(height, width, channels)

        # globals(background)
        # globals(foreground1)
        # globals(foreground2)
        # globals(foreground3)
        # globals(base_name)
        background = Image.new('RGB', (height, width), color='black').convert('RGBA')
        # print(filename)
        base_name = os.path.basename(filename)
        # print(base_name)
        # print(base_name.split('_')[0])

        if filename.__contains__('caolv'):
            print(base_name)
            foreground1 = Image.open(filename).convert('RGBA')
            # background.paste(foreground1, (0, 0), foreground1)
            counter = counter + 1
        if filename.__contains__('fen'):
            print(base_name)
            foreground2 = Image.open(filename).convert('RGBA')
            # background.paste(foreground2, (0, 0), foreground2)
            counter = counter + 1
        if filename.__contains__('lightblue'):
            print(base_name)
            foreground3 = Image.open(filename).convert('RGBA')
            # background.paste(foreground3, (0, 0), foreground3)
            counter = counter + 1


        # background.paste(foreground1, (0, 0), foreground1)
        # background.paste(foreground2, (0, 0), foreground2)
        # background.paste(foreground3, (0, 0), foreground3)

        # background.save("/home/luolu/PycharmProjects/color_detection/data/result_qaak/result/"
        #                     + base_name.split('_')[0] + ".png")
        print('counter:', counter)
        if counter == 3:
            background.paste(foreground1, (0, 0), foreground1)
            background.paste(foreground2, (0, 0), foreground2)
            background.paste(foreground3, (0, 0), foreground3)
            background.save("/home/luolu/PycharmProjects/color_detection/data/result_qaak/result/"
                            + base_name.split('_')[0] + ".png")
            counter = 0
