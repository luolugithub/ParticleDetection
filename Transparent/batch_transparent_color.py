# -*- coding: utf-8 -*-
# @Time : 2020/3/6 上午10:35
# @Author : LuoLu
# @FileName: color_img_transparent.py
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

# caolv     > 150
# lb        > 150
# fen       < 150


if __name__ == '__main__':
    for filename in glob.glob('/home/luolu/PycharmProjects/color_detection/data/result_qaak/caolv_mask/*.png'):
        img = cv.imread(filename)
        # print(filename)
        base_name = os.path.basename(filename)
        print(base_name)
        img = Image.open(filename)
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                if item[0] > 150:
                    newData.append((0, 0, 0, 255))
                else:
                    newData.append(item)
                    print(item)

        img.putdata(newData)
        img.save("/home/luolu/PycharmProjects/color_detection/data/result_qaak/transparent/" + base_name)