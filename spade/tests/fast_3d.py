import sys

from skimage.io import imread
import numpy as np

from spade.detection_3d import spade3d
from spade.shapes.library import potatoids_5x5__smallest_1_pix
from spade.data_binding import *


image = imread('supereasy.tif')

result = spade3d(image, potatoids_5x5__smallest_1_pix,
                 threshold=255,
                 minimal_z_thickness=2,
                 data_binding_function=None)

expected = np.array([[[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]])
print(result)

if (result == expected).all():
    print("*************\nTODOCH BIENCH\n*************")
    sys.exit(0)

print('PROBLEM')
sys.exit(1)
