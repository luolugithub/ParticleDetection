from spade_calibration.tools.scores import SpadeScores
from spade.detection_2d import spade2d
from spade.data_binding import *
from skimage.io import imread
from spade.shapes.examples import *

import numpy as np
import sys


image = imread('supereasy.tif')[0, :, :]

mask = np.ones_like(image, bool)

expected = np.array([[0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0]])

scores = SpadeScores(image, mask, potatoids5x5_smallest1px_thickrings,
                     mean_difference)
scores.reduce_array(4)
result = scores.get_result(100, True, 1)
result2 = spade2d(image, potatoids5x5_smallest4px_thickrings, 100)

print(result[0])
print(result2)

if (result[0] == expected).all() and (result2 == expected).all():
    print("*************\nTODOCH BIENCH\n*************")
    sys.exit(0)

print('PROBLEM')
sys.exit(1)
