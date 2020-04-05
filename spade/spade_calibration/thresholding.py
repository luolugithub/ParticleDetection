import sys
import os
import os.path

import numpy as np
from skimage.io import imread
from skimage.morphology import remove_small_objects
from skimage.measure import label

from spade_calibration.tools.expertise import read_expertise, \
    compare_to_expertise, mip_expertise
from spade_calibration.tools.io import load
from spade_calibration.preprocessing import *

preprocessing_funcs = [normalize_mean_maximum_intensity_projection,
                       median_filter]

output_folder = '/home/ncedilni/simple_thresholding_results_new/'
os.makedirs(output_folder, exist_ok=True)

image_full_path = sys.argv[1]
calibration_data_path = os.path.abspath(
    os.path.join(os.path.dirname(image_full_path), os.pardir))
image_name = os.path.basename(image_full_path)

original_image = imread(image_full_path)
original_mask = load(os.path.join(calibration_data_path, 'masks', image_name))
expertise = read_expertise(os.path.join(calibration_data_path, 'expertise',
                                        image_name + '.txt'))

output_csvfile = os.path.join(output_folder, '{}_97.csv'.format(image_name))

with open(output_csvfile, 'w') as file_handler:
    file_handler.write('preprocessing,data_binding,centile,'
                       'minimal_shape_surface,threshold,'
                       'minimal_surface_3d,minimal_z_thickness,'
                       'minimal_z_overlap,tp,fp,fn,dd\n')
    for preprocessing_func in preprocessing_funcs:
        image, mask = preprocessing_func(original_image, original_mask)
        if 'maximum_intensity_projection' in preprocessing_func.__name__:
            corrected_expertise = mip_expertise(expertise)
        else:
            corrected_expertise = expertise
        for threshold in np.linspace(image[mask].min(),
                                     image[mask].max(),
                                     2000):
            binary_image = image > threshold
            for minimal_volume in range(1, 5):
                cleaned_binary_image = remove_small_objects(binary_image,
                                                            minimal_volume)
                labeled_image = label(cleaned_binary_image)
                tp, fp, fn, dd = compare_to_expertise(corrected_expertise,
                                                      labeled_image)
                file_handler.write(
                    '{},simple_thresholding,0,{},{},0,0,0,{},{},{},'
                    '{}\n'.format(
                        preprocessing_func.__name__, minimal_volume,
                        threshold, tp, fp, fn, dd))
