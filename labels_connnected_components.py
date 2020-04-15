"""
Labelling connected components of an image
===========================================

This example shows how to label connected components of a binary image, using
the dedicated skimage.measure.label function.
"""
import cv2
import matplotlib
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np

n = 12
l = 256
np.random.seed(1)
# im = np.zeros((l, l))
# points = l * np.random.random((2, n ** 2))
# im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1

matplotlib.use('TkAgg')

path = "/home/luolu/PycharmProjects/ParticleDetection/data/yashi_qscan/edge/edge_cl.png"

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

im = filters.gaussian(gray)
blobs = im > 0.7 * im.mean()

all_labels = measure.label(blobs)
blobs_labels = measure.label(blobs, background=0)

plt.figure(figsize=(9, 9))
plt.subplot(221)
plt.imshow(blobs, cmap='gray')
plt.axis('off')
plt.subplot(222)
plt.imshow(all_labels, cmap='nipy_spectral')
plt.axis('off')
plt.subplot(223)
plt.imshow(blobs_labels, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()
