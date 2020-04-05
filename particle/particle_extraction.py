# -*- coding: utf-8 -*-
# @Time : 2020/3/20 上午9:33
# @Author : LuoLu
# @FileName: particle_extraction.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

from skimage.io import imread, imshow
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
from numpy import percentile
import matplotlib.pyplot as plt

# from spade.spade.detection_2d import spade2d
from spade.spade.shapes.examples import potatoids5x5_smallest3px

# Load the example image.
from spade.spade.detection_2d import spade2d

image = imread("/home/luolu/PycharmProjects/ParticleDetection/35_mask_caolv.png")

# Separate cell image from background, using by Otsu's thresholding method.
cell = image > threshold_otsu(image)

# Focus on brightest pixels only
potential_centers = image > percentile(image[cell], 99)

# Detect particles.
particles = spade2d(image=image,
                    shapes_library=potatoids5x5_smallest3px,
                    threshold=20,
                    potential_centers=potential_centers,
                    mask=cell)

# Show detected particles as overlay on our original image.
imshow(label2rgb(particles, image, bg_label=0))
plt.show()
