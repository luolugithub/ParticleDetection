# -*- coding: utf-8 -*-
# @Time : 2020/4/15 上午10:38
# @Author : LuoLu
# @FileName: length_contact.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import numpy as np
import cv2 as cv

img = cv.imread('/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/contact_test.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
num_labels, labels_im = cv.connectedComponents(thresh)
print("num_labels:", num_labels - 1)
print("len contours :", len(contours))
count0 = 0

dots = 0
line = 0

for iterm in range(len(contours)):
    len_contact = cv.arcLength(contours[iterm], False)
    print("iterm:" + str(iterm) + "," + "perimeter = " + str(len_contact))
    if len_contact != 0:
        if len_contact > 30:
            line = line + 1
        else:
            dots = dots + 1
# print("count0 contours :", count0)
# print("jian :", len(contours) - count0)
print("dots :", dots)
print("line :", line)
print(type(img))
img = np.copy(img)
cv.putText(img, "dots Sum:" + str(dots), (30, 960), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
cv.putText(img, "line Sum:" + str(line), (30, 990), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
cv.imwrite("/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/perimeter/result_contact.png", img)
cv.imshow("img", img)
cv.waitKey()
cv.destroyAllWindows()