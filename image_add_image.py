# -*- coding: utf-8 -*-
# @Time : 2020/4/2 下午4:51
# @Author : LuoLu
# @FileName: image_add_image.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import cv2
import numpy as np

img1 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/background_line_pills.png')
img2 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/distance_pills_02.png')
img3 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/thresh_pills_02.png')
img4 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/sure_bg_thresh_pills_02.png')
img5 = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/red_contour_pills_02.png')

gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
ret1, thresh_img1 = cv2.threshold(gray_img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret2, thresh_img2 = cv2.threshold(gray_img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# dst = cv2.add(img1, img2)
subtract = cv2.subtract(img1, img2)
add = cv2.add(img1, img2)
bitwise_xor = cv2.bitwise_not(gray_img1, gray_img2)
bitwise_and = cv2.add(img1, img4)
compare12 = np.concatenate((img1, img2), axis=1) #side by side comparison
compare13 = np.concatenate((img1, img3), axis=1) #side by side comparison
compare23 = np.concatenate((img2, img3), axis=1) #side by side comparison
compare24 = np.concatenate((img2, img4), axis=1) #side by side comparison
compare25 = np.concatenate((img5, bitwise_and), axis=1) #side by side comparison
# subtract_compare = np.concatenate((subtract, img2), axis=1) #side by side comparison
# add_compare = np.concatenate((add, img2), axis=1) #side by side comparison
# bitwiseAnd = cv2.bitwise_not(img1, img2)
# bitwiseAnd_thresh = cv2.subtract(bitwiseAnd, img3)
# add_subtract = cv2.bitwise_not(subtract, add)
# # cv2.imshow('dst', dst)
cv2.imshow('bitwise_and', bitwise_and)
# cv2.imshow('add', add)
# cv2.imshow('compare12', compare12)
# cv2.imshow('compare13', compare13)
# cv2.imshow('compare23', compare23)
cv2.imshow('compare14', compare24)
cv2.imshow('compare25', compare25)
# cv2.imshow('compare', compare)
# cv2.imshow('subtract_compare', subtract_compare)
# cv2.imshow('add_compare', add_compare)
# cv2.imshow('bitwiseAnd', bitwiseAnd)
# cv2.imshow('bitwiseAnd_thresh', bitwiseAnd_thresh)
# cv2.imshow('bitwise_not', bitwise_and)
cv2.waitKey(0)
cv2.destroyAllWindows()
