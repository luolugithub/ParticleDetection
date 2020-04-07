# -*- coding: utf-8 -*-
# @Time : 2020/4/5 下午4:34
# @Author : LuoLu
# @FileName: logic_relation2binary.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2
import numpy as np

img1 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/binary_ct.png', 0)
img2 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/wline_ct.png', 0)

img_bwa = cv2.bitwise_and(img1, img2)
img_bwo = cv2.bitwise_or(img1, img2)
img_bwx = cv2.bitwise_xor(img1, img2)

compare12 = np.concatenate((img1, img2), axis=1)
compare1bwa = np.concatenate((img1, img_bwa), axis=1)

cv2.imshow("binary image & watershed line", compare12)
cv2.imshow("src & touch result", compare1bwa)
cv2.imshow("Bitwise AND of Image 1 and 2", img_bwa)
cv2.imshow("Bitwise OR of Image 1 and 2", img_bwo)
cv2.imshow("Bitwise XOR of Image 1 and 2", img_bwx)
cv2.waitKey(0)
cv2.destroyAllWindows()
