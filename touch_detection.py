# -*- coding: utf-8 -*-
# @Time : 2020/3/24 上午11:02
# @Author : LuoLu
# @FileName: touch_detection.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import cv2

im = cv2.imread('/data/mask/close_2_mask_fen.png')
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
borderSize = 75
distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
gap = 10
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                             cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
mn, mx, _, _ = cv2.minMaxLoc(nxcor)
th, peaks = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
peaks8u = cv2.convertScaleAbs(peaks)
contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
peaks8u = cv2.convertScaleAbs(peaks)  # to use as mask
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y + h, x:x + w], peaks8u[y:y + h, x:x + w])
    cv2.circle(im, (int(mxloc[0] + x), int(mxloc[1] + y)), int(mx), (255, 0, 0), 2)
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.drawContours(im, contours, i, (0, 0, 255), 2)

cv2.imshow('circles', im)
cv2.waitKey(0)
