# -*- coding: utf-8 -*-
# @Time : 2020/3/5 上午9:46
# @Author : LuoLu
# @FileName: single_color_replace.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

from PIL import Image
import numpy as np

im = Image.open('data/image/trans_line_thresh_pills_02.png')
im = im.convert('RGBA')

data = np.array(im)  # "data" is a height x width x 4 numpy array
print(data.shape)
red, green, blue, alpha = data.T  # Temporarily unpack the bands for readability

# Replace white with red... (leaves alpha values alone...)
white_areas = (red == 255) & (blue == 255) & (green == 255)
data[..., :-1][white_areas.T] = (0, 0, 0)  # Transpose back needed

# Replace other with black...
# white_areas = (red == 255) & (blue == 255) & (green == 255)
# if white_areas != (red == 255) & (blue == 192) & (green == 192):
#     data[..., :-1][white_areas.T] = (255, 255, 255)
#
#
im2 = Image.fromarray(data)
im2.save("data/image//trans_line_black_pills_02.png")
im2.show()
