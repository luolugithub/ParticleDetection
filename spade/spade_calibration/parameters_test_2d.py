import sys
import os.path
from os import makedirs
from configparser import ConfigParser

from numpy import percentile, vstack, array, arange, nanmax, append
from skimage.io import imread

from spade.data_binding import *
from spade.shapes.examples import *
from spade_calibration.preprocessing import *
from spade_calibration.tools.expertise import read_expertise, \
    compare_to_expertise, mip_expertise
from spade_calibration.tools.scores import SpadeScores
from spade_calibration.tools.io import load, printnow


config = ConfigParser()
config.read('calibration.ini')
output_folder = config['general']['output_folder']
makedirs(output_folder, exist_ok=True)
exec("preprocessing_functions = [{}]".format(
    config['general']['preprocessing_functions']))
exec("data_binding_functions = [{}]".format(
    config['general']['data_binding_functions']))
exec("shape_libraries = [{}]".format(
    config['general']['shape_libraries']))
minimal_shape_surfaces = arange(
    config.getint('minimal_shape_surfaces', 'start'),
    config.getint('minimal_shape_surfaces', 'stop'),
    config.getint('minimal_shape_surfaces', 'step'))
centiles = arange(
    config.getfloat('centiles', 'start'),
    config.getfloat('centiles', 'stop'),
    config.getfloat('centiles', 'step'))
number_of_thresholds = config.getint('limits', 'number_of_thresholds')
max_candidates = config.getint('limits', 'maximum_z_candidates')

image_full_path = sys.argv[1]
calibration_data_path = os.path.abspath(
    os.path.join(os.path.dirname(image_full_path), os.pardir))
image_name = os.path.basename(image_full_path)

image = imread(image_full_path)
mask = load(os.path.join(calibration_data_path, 'masks', image_name))
expertise = read_expertise(os.path.join(calibration_data_path, 'expertise',
                                        image_name + '.txt'))

i = 0
output_csvfile = os.path.join(output_folder, '{}_{}.csv'.format(image_name, i))
while os.path.exists(output_csvfile):
    i += 1
    output_csvfile = os.path.join(output_folder, '{}_{}.csv'.format(image_name,
                                                                    i))

corrected_expertise = mip_expertise(expertise)


# TODO: make scores for 3D images more understandable. other class?
printnow('Parameters tests for image ', image_name)
with open(output_csvfile, 'w') as file_handler:
    file_handler.write('preprocessing,centile,library,'
                       'minsurf,data_binding,threshold,'
                       'tp,fp,fn,dd\n')
    for preprocessing_function in preprocessing_functions:
        printnow('Preprocessing image with ',
                 preprocessing_function.__name__, '...')
        preprocessed_image, preprocessed_mask = preprocessing_function(image,
                                                                       mask)
        for data_binding_function in data_binding_functions:
            printnow('Computing scores with ',
                     data_binding_function.__name__, '...')
            minimal_threshold = config.getfloat(
                "limits", data_binding_function.__name__)
            for shape_library in shape_libraries:
                printnow('Using library ', shape_library.name, '...')
                scores = SpadeScores(preprocessed_image[0],
                                     preprocessed_mask[0],
                                     shape_library,
                                     data_binding_function)
                for centile in centiles:
                    printnow('Starting tests at centile ', centile, '...')
                    exclude_pixels = preprocessed_image <= percentile(
                        preprocessed_image[preprocessed_mask], centile)
                    for minimal_shape_surface in minimal_shape_surfaces:
                        scores.reduce_array(minimal_shape_surface,
                                            exclude_pixels[0])
                        maximum_score = scores.get_max_score()
                        step = (maximum_score - minimal_threshold) / \
                                               number_of_thresholds
                        threshold = maximum_score
                        while threshold > minimal_threshold:
                            previous_number_of_objects = 1
                            result = scores.get_result(threshold, True)
                            labeled_image_pseudo3d = array([result[0]])
                            candidates = result[1]
                            tp, fp, fn, dd = compare_to_expertise(
                                corrected_expertise, labeled_image_pseudo3d)
                            file_handler.write(
                                '{},{},{},{},{},{},{},{},{},{}\n'
                                    .format(
                                    preprocessing_function.__name__,
                                    centile,
                                    shape_library.name,
                                    minimal_shape_surface,
                                    data_binding_function.__name__,
                                    threshold,
                                    tp, fp, fn, dd))
                            if len(candidates) > max_candidates:
                                break
                            threshold -= step
printnow('Done!')
