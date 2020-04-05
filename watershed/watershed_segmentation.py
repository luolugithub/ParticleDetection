# -*- coding: utf-8 -*-
# @Time : 2020/4/5 下午3:55
# @Author : LuoLu
# @FileName: watershed_segmentation.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_local
from PIL import Image

img = cv2.imread('/home/luolu/PycharmProjects/ParticleDetection/data/image/thresh_pills_02.png', 0)

'''Adaptive thersholding 
   calculates thresholds in regions of size block_size surrounding each pixel
   to handle the non-uniform background'''
block_size = 51
adaptive_thresh = threshold_local(img, block_size)  # , offset=10)
binary_adaptive = img > adaptive_thresh
# cv2.imwrite('/home/luolu/PycharmProjects/ParticleDetection/data/image/binary_37.png', Image.fromarray(binary_adaptive))

# Calculate Euclidean distance
distance = ndi.distance_transform_edt(binary_adaptive)

# Find local maxima of the distance map
local_maxi = peak_local_max(distance, labels=binary_adaptive, footprint=np.ones((51, 51)), indices=False)
# Label the maxima
markers = ndi.label(local_maxi)[0]

''' Watershed algorithm
    The option watershed_line=True leave a one-pixel wide line 
    with label 0 separating the regions obtained by the watershed algorithm '''
labels = watershed(-distance, markers, watershed_line=True)

# Plot the result
print(type(labels))
print(labels.shape)
plt.imshow(img, cmap='gray')
plt.imshow(labels == 0, alpha=0.3, cmap='Reds')
plt.show()
# cv2.imwrite('/home/luolu/PycharmProjects/ParticleDetection/data/image/wline_pills_02', labels)

img = Image.fromarray(labels == 0)
# img.save('/home/luolu/PycharmProjects/ParticleDetection/data/image/wline_37.png')
