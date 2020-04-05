# -*- coding: utf-8 -*-
# @Time : 2020/4/3 下午2:06
# @Author : LuoLu
# @FileName: extract_watershed_contours.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com
import cv2
import numpy as np
from PIL import Image
from Watershed import *

# image = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/thresh_pills_02.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# # cv2.imwrite("/home/luolu/PycharmProjects/ParticleDetection/data/image/thresh_pills_02.png", thresh)
# # noise removal
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#
# sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
# sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
# unknown = cv2.add(sure_bg, sure_fg)  # unknown area
# # cv2.imshow('unknown', unknown)
#
# # Perform the distance transform algorithm
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L1, 5)
# # Normalize the distance image for range = {0.0, 1.0}
# cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
#
# # Finding sure foreground area
# ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)
#
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers + 1
# # Now, mark the region of unknown with zero
# markers[unknown == 255] = 0
#
# markers_copy = markers.copy()
# markers_copy[markers == 0] = 255  # 灰色表示背景
# markers_copy[markers == 1] = 0  # 黑色表示背景
# markers_copy[markers > 1] = 255  # 白色表示前景
#
# markers_copy = np.uint8(markers_copy)

# 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
shed = Watershed(
    data_image="/home/luolu/PycharmProjects/ParticleDetection/data/image/particle.png",
    binary_or_gray_or_color="color",
    size_for_calculations=920,
    sigma=1,
    gradient_threshold_as_fraction=0.1,
    level_decimation_factor=16,
    padding=20)
shed.extract_data_pixels()
print("extract_data_pixels")
shed.display_data_image()
shed.mark_image_regions_for_gradient_mods()  # (A)
shed.compute_gradient_image()
print("compute_gradient_image")
shed.modify_gradients_with_marker_minima()  # (B)
shed.compute_Z_level_sets_for_gradient_image()
print("compute_Z_level_sets_for_gradient_image")
shed.propagate_influence_zones_from_bottom_to_top_of_Z_levels()
shed.display_watershed()
print("display_watershed")
# shed.display_watershed_in_color()
shed.extract_watershed_contours_separated()
shed.display_watershed_contours_in_color()
print("extract_watershed_contours_separated")
shed.extract_watershed_contours_with_random_sampling(1, 25)  # (C)
print("display_watershed_contours_in_color")
shed.extract_segmented_blobs_using_contours()
# shed.display_all_segmented_blobs()

# cv2.imshow('unknown_opening_thresh', unknown_opening_thresh)
cv2.waitKey()
