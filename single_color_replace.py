# -*- coding: utf-8 -*-
# @Time : 2020/3/5 上午9:46
# @Author : LuoLu
# @FileName: single_color_replace.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

from PIL import Image
import numpy as np

im = Image.open('data/yashi_qscan/cleaned/lb.png')
im = im.convert('RGBA')

data = np.array(im)  # "data" is a height x width x 4 numpy array
print(data.shape)
red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

# Replace white with red... (leaves alpha values alone...)
white_areas = (red == 255) & (blue == 255) & (green == 255)
data[..., :-1][white_areas.T] = (0, 255, 255)  # Transpose back needed

# light Blue 0, 255, 255
# fen,  255, 192, 192
# caolv, 0, 128, 128
im2 = Image.fromarray(data)
im2.save("data/yashi_qscan/color/lb_color.png")
im2.show()
