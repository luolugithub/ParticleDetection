# -*- coding: utf-8 -*-
# @Time    : 2019/12/25 上午9:10
# @Author  : luolu
# @Email   : argluolu@gmail.com
# @File    : generate_white_img.py
# @Software: PyCharm

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm


img = Image.new('RGB', (1001, 1001), color=(147, 148, 146))
d = ImageDraw.Draw(img)
root = tk.Tk()
# get a font
fonts = list(set([f.name for f in fm.fontManager.ttflist]))
fonts.sort()
combo = ttk.Combobox(root, value=fonts)
combo.pack()
font = ImageFont.truetype('/home/luolu/PycharmProjects/ParticleDetection/bmjs.ttf', 42)
# font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/ubuntu-B.ttf', 28)
d.text((250, 300), text="dots sum = 431", fill=(0, 0, 0), font=font)
d.text((250, 350), text="line sum = 84", fill=(0, 0, 0), font=font)
d.text((250, 400), text="particle sum = 196", fill=(0, 0, 0), font=font)
d.text((250, 450), text="perimeter sum = 56262", fill=(0, 0, 0), font=font)
# 129.3 + 58.8 //196
d.text((50, 500), text="formula = (dots*0.3 + line*0.7)/particle sum", fill=(0, 0, 0), font=font)
d.text((250, 550), text="index = 0.95969387", fill=(0, 0, 0), font=font)
img.save('/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/perimeter/result.png', format="PNG")
