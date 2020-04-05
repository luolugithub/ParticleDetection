import sys
import os.path
from configparser import ConfigParser

from numpy import percentile, vstack, array, arange, nanmax, append
from skimage.io import imread

from spade.data_binding import *
from spade.shapes.library import potatoids_5x5__smallest_1_pix
from spade.detection_3d import combine_2d_shapes
from spade_calibration.preprocessing import *
from spade_calibration.tools.expertise import read_expertise, \
    compare_to_expertise, mip_expertise
from spade_calibration.tools.scores import SpadeScores
from spade_calibration.tools.io import load, printnow


config = ConfigParser()
config.read('calibration.ini')
output_folder = config['general']['output_folder']
exec("preprocessing_functions = [{}]".format(
    config['general']['preprocessing_functions']))
exec("data_binding_functions = [{}]".format(
    config['general']['data_binding_functions']))
minimal_shape_surfaces = arange(
    config.getint('minimal_shape_surfaces', 'start'),
    config.getint('minimal_shape_surfaces', 'stop'),
    config.getint('minimal_shape_surfaces', 'step'))
minimal_surface_3ds = arange(
    config.getint('minimal_surface_3ds', 'start'),
    config.getint('minimal_surface_3ds', 'stop'),
    config.getint('minimal_surface_3ds', 'step'))
minimal_z_thicknesses = arange(
    config.getint('minimal_z_thicknesses', 'start'),
    config.getint('minimal_z_thicknesses', 'stop'),
    config.getint('minimal_z_thicknesses', 'step'))
minimal_z_overlaps = arange(
    config.getfloat('minimal_z_overlaps', 'start'),
    config.getfloat('minimal_z_overlaps', 'stop'),
    config.getfloat('minimal_z_overlaps', 'step'))
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

# TODO: make scores for 3D images more understandable. other class?
printnow('Parameters tests for image ', image_name)
with open(output_csvfile, 'w') as file_handler:
    file_handler.write('preprocessing,data_binding,centile,'
                       'minimal_shape_surface,threshold,minimal_surface_3d,'
                       'minimal_z_thickness,minimal_z_overlap,'
                       'tp,fp,fn,dd\n')
    for preprocessing_function in preprocessing_functions:
        printnow('Preprocessing image with ',
                 preprocessing_function.__name__, '...')
        preprocessed_image, preprocessed_mask = preprocessing_function(image,
                                                                       mask)
        if 'maximum_intensity_projection' in preprocessing_function.__name__:
            corrected_expertise = mip_expertise(expertise)
            corrected_max_candidates = max_candidates / 3
        else:
            corrected_expertise = expertise
            corrected_max_candidates = max_candidates
        for data_binding_function in data_binding_functions:
            printnow('Computing scores with ',
                     data_binding_function.__name__, '...')
            minimal_threshold = config.getfloat(
                "limits", data_binding_function.__name__)
            scores_3d = [SpadeScores(image_slice,
                                     preprocessed_mask[z],
                                     potatoids_5x5__smallest_1_pix,
                                     data_binding_function)
                         for z, image_slice in enumerate(preprocessed_image)]
            for centile in centiles:
                printnow('Starting tests at centile ', centile, '...')
                exclude_pixels = preprocessed_image <= percentile(
                    preprocessed_image[preprocessed_mask], centile)
                for minimal_shape_surface in minimal_shape_surfaces:
                    for z, scores_2d in enumerate(scores_3d):
                        scores_2d.reduce_array(minimal_shape_surface,
                                               exclude_pixels[z])
                    maximum_score = nanmax(
                        [scores_2d.get_max_score() for scores_2d in scores_3d])
                    step = (maximum_score - minimal_threshold) / \
                                           number_of_thresholds
                    threshold = maximum_score
                    while threshold > minimal_threshold:
                        previous_number_of_objects = 1
                        results_2d = []
                        for scores_2d in scores_3d:
                            results_2d += [scores_2d.get_result(
                                threshold,
                                True,
                                previous_number_of_objects)]
                            previous_number_of_objects += len(
                                results_2d[-1][1])
                        labeled_image_pseudo3d = array(
                            [result[0] for result in results_2d])
                        candidates = vstack(
                            [append([[z]] * result[1].shape[0],
                                    result[1],
                                    axis=1)
                             for z, result in enumerate(results_2d)
                             if len(result[1]) > 0])
                        if len(candidates) > corrected_max_candidates:
                            break
                        if 'maximum_intensity_projection' in \
                                preprocessing_function.__name__:
                            tp, fp, fn, dd = compare_to_expertise(
                                corrected_expertise, labeled_image_pseudo3d)
                            file_handler.write(
                                '{},{},{},{},{},{},{},{},{},{},{},{}\n'
                                    .format(
                                    preprocessing_function.__name__,
                                    data_binding_function.__name__,
                                    centile,
                                    minimal_shape_surface,
                                    threshold,
                                    0,
                                    0,
                                    0,
                                    tp, fp, fn, dd))
                        else:
                            for minimal_surface_3d in minimal_surface_3ds:
                                if minimal_surface_3d < minimal_shape_surface:
                                    continue
                                for minimal_z_thickness in \
                                        minimal_z_thicknesses:
                                    for minimal_z_overlap in \
                                            minimal_z_overlaps:
                                        labeled_image_3d = combine_2d_shapes(
                                            labeled_image_pseudo3d,
                                            candidates,
                                            scores_3d[
                                                0].reduced_shapes_library,
                                            minimal_surface_3d,
                                            minimal_z_thickness,
                                            minimal_z_overlap)
                                        tp, fp, fn, dd = compare_to_expertise(
                                            corrected_expertise,
                                            labeled_image_3d)
                                        file_handler.write(
                                            ('{},{},{},{},{},{},{}' +
                                             ',{},{},{},{},{}\n').format(
                                            preprocessing_function.__name__,
                                            data_binding_function.__name__,
                                            centile,
                                            minimal_shape_surface,
                                            threshold,
                                            minimal_surface_3d,
                                            minimal_z_thickness,
                                            minimal_z_overlap,
                                            tp, fp, fn, dd))
                        threshold -= step
printnow('Done!')
