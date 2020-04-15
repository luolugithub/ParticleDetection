# -*- coding: utf-8 -*-
# @Time : 2020/4/15 上午9:00
# @Author : LuoLu
# @FileName: hsv_split.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

image = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/color/lb_color.png',
                   cv2.IMREAD_UNCHANGED)
edge_img = cv2.imread("/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/edge/edge_lb.png",
                      cv2.IMREAD_UNCHANGED)

# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsv = cv2.split(hsv)
# gray = hsv[0]
src = cv2.GaussianBlur(image, (1, 1), 0)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
cv2.imshow("binary", image)

# contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
colors = []
for i in range(num_labels):
    b = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    r = np.random.randint(0, 256)
    colors.append((b, g, r))
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(type(contours))
print("len contours:", len(contours))
print("num_labels:", num_labels)
colors[0] = (0, 0, 0)
cv2.drawContours(image, contours, -1, (107, 61, 88), thickness=2)
image = np.copy(image)


class Perimeter:
    sum_perimeter = 0  # Access through class

instance_perimeter = Perimeter()
for iterm in range(1, num_labels, 1):
    x, y, w, h, area = stats[iterm]
    cx, cy = centers[iterm]
    # 画出外接矩形
    cv2.rectangle(image, (x, y), (x + w, y + h), colors[iterm], 1, 8, 0)
    cv2.putText(image, str(iterm),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5, (0, 0, 255),
                1)
    print("label index %d, area of the label : %d" % (iterm, area))
    cv2.imshow("colored labels", image)
    # cv.imwrite("labels.png", image)
    print("total number : ", num_labels - 1)
    # cv2.putText(image, "p Sum:" + str(instance_perimeter.sum_perimeter), (30, 900), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 1)

for l in range(len(contours)):
    perimeter = cv2.arcLength(contours[l], True)
    instance_perimeter.sum_perimeter = instance_perimeter.sum_perimeter + round(perimeter)
    print("iterm:" + str(l) + "," + "l=" + str(perimeter))

print("total sum_perimeter : ", instance_perimeter.sum_perimeter)


image = np.copy(image)
cv2.putText(image, "particle Sum:" + str(num_labels - 1), (30, 960), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
cv2.putText(image, "perimeter Sum:" + str(instance_perimeter.sum_perimeter), (30, 990), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
cv2.imwrite("/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/perimeter/lb.png", image)

# print("total sum_perimeter : ", sum_perimeter)
# cv2.drawContours(image, contours, -1, (255, 0, 0), thickness=2)
# cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
