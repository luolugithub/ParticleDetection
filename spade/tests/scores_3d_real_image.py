import os.path

import numpy as np
from skimage.io import imread

from spade_calibration.tools.io import load, imsave
from spade_calibration.tools.expertise import read_expertise, \
    compare_to_expertise
from spade_calibration.tools.scores import SpadeScores
from spade_calibration.preprocessing import *
from spade.data_binding import *
from spade.detection_3d import combine_2d_shapes
from spade.shapes.library import potatoids_5x5__smallest_1_pix

image_full_path = "/home/nicoco/Documents/Dropbox/spade/calibration_data" \
                  "/images/410T1_B03_S09_W2_cell3_ann.tif"

image_name = os.path.basename(image_full_path)
calibration_data_path = os.path.abspath(
    os.path.join(os.path.dirname(image_full_path), os.pardir))
image = imread(image_full_path)
mask = load(os.path.join(calibration_data_path, 'masks', image_name))
expertise = read_expertise(os.path.join(calibration_data_path, 'expertise',
                                        image_name + '.txt'))

preprocessed_image, preprocessed_mask = normalize_variance(image, mask)

scores_3d = [SpadeScores(image_slice,
                         preprocessed_mask[z],
                         potatoids_5x5__smallest_1_pix,
                         mean_difference)
             for z, image_slice in enumerate(preprocessed_image)]
maximum_score = np.nanmax(
    [scores_2d.get_max_score() for scores_2d in scores_3d])

print(maximum_score)
previous_number_of_objects = 1
results_2d = []

for scores_2d in scores_3d:
    results_2d += [scores_2d.get_result(maximum_score/1.5,
                                        True,
                                        previous_number_of_objects)]
    previous_number_of_objects += len(results_2d[-1][1])



labeled_image_pseudo3d = np.array([result[0] for result in results_2d])
candidates = np.vstack([np.append([[z]] * result[1].shape[0], result[1],
                                  axis=1)
                        for z, result in enumerate(results_2d)
                        if len(result[1]) > 0])
labeled_image_3d = combine_2d_shapes(
    labeled_image_pseudo3d, candidates, scores_3d[0].shapes_library, 3, 3, 0.5)

tp, fp, fn, dd, expmask = compare_to_expertise(expertise, labeled_image_3d,
                                              True)

print(tp, fp, fn, dd)
imsave('/home/3dtest.tif', image, labeled_image_pseudo3d, expmask, mask)