# -*- coding: utf-8 -*-
# @Time : 2020/4/8 上午10:52
# @Author : LuoLu
# @FileName: conncted_components.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2
import numpy as np
from PIL import Image
from collections import Counter

img = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/edge/edge_cl.png', 0)
img = cv2.GaussianBlur(img, (1, 1), 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
num_labels, labels_im = cv2.connectedComponents(img)
print("num_labels:", num_labels - 1)
# Image._show(Image.fromarray(labels_im == 2), title="labels_im.jpg")
print("len labels_im:", (labels_im == 2).shape)
print("type labels_im:", type(labels_im))
# print("len 1:", np.where((labels_im == 2).sum()))
# count = [0] * (num_labels)
# print("count len:", range(num_labels))
# for num in range(1, num_labels, 1):
#     for i in range((labels_im == num).shape[0]):
#         for j in range((labels_im == num).shape[1]):
#             if (labels_im == num)[i][j] == 1:
#                 count[num] = count[num] + 1
#
# print("count len:", len(count))
# for iterm in range(1, len(count), 1):
#     print("iterm" + str(iterm) + ":", count[iterm])





def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    # Image._show(Image.fromarray(label_hue), title="label_hue.jpg")

    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # Image._show(Image.fromarray(blank_ch), title="blank_ch.jpg")

    # set bg label to black
    labeled_img[label_hue == 0] = 40

    cv2.namedWindow("labeled", flags=2)
    cv2.imshow("labeled", labeled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imshow_components(labels_im)
