# TODO: clean outputs and variable names
# TODO: split into several files
import sys
import os
import os.path
import math
from copy import deepcopy
from argparse import ArgumentParser
from multiprocessing import cpu_count, Process, Queue

from skimage.io import imread
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from spade_calibration.preprocessing import *
from spade_calibration.tools.io import *
from spade_calibration.tools.expertise import *
from spade_calibration.tools.statistics import *


def log_function(x, p1, p2):
    return p1 + p2 * np.log(x)


def sqrt_function(x, p1, p2):
    return p1 + p2 * np.sqrt(x)


def inverse_function(x, p1, p2):
    return p1 - p2 / x


def polynomial_function(x, *params):
    return sum([p * (x ** i) for i, p in enumerate(params)])


def find_nearest_sorted_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx == len(array) or \
                    math.fabs(value - array[idx - 1]) < math.fabs(
                        value - array[idx]):
        return idx - 1
    else:
        return idx


def find_nearest_sorted(array, value):
    return array[find_nearest_sorted_idx(array, value)]


def total_variation_3d(array, mask):
    if array.shape[0] > 1:
        gradients = np.gradient(array)
        return np.sum([gradient[mask].sum() for gradient in gradients])
    else:  # MIP FIX
        return total_variation_2d(array, mask)


def total_variation_2d(array, mask):
    if array.shape[0] > 1:
        gradients = np.gradient(array)
        return np.sum([gradient[mask].sum() for gradient in gradients[1:]])
    else:  # MIP FIX
        gradients = np.gradient(array[0])
        return np.sum([gradient[mask[0]].sum() for gradient in gradients])


def get_image_statistics(image, mask, preprocessing_function_name=''):
    return {'mean' + preprocessing_function_name: image[mask].mean(),
            'median' + preprocessing_function_name: np.median(image[mask]),
            'max' + preprocessing_function_name: np.max(image[mask]),
            'var' + preprocessing_function_name: np.var(image[mask]),
            'total_var2d' + preprocessing_function_name: total_variation_2d(
                image, mask),
            'total_var3d' + preprocessing_function_name: total_variation_3d(
                image, mask)}


def get_images_statistics(list_of_image_names):
    result = []
    for image_name in list_of_image_names:
        image = imread('../../calibration_data/images/' + image_name)
        mask = load('../../calibration_data/masks/' + image_name)
        particles = load('../../blind_results/ideal/' + image_name +
                         '_particles')
        cheat_mask = np.logical_xor(mask, particles.astype(bool))
        result += [{'name': image_name}]
        result[-1]['cheat var'] = np.var(image[cheat_mask])
        result[-1].update(get_image_statistics(image, mask))
        for preprocessing_function in preprocessing_functions:
            preprocessed_image, preprocessed_mask = preprocessing_function(
                image, mask)
            result[-1].update(get_image_statistics(
                preprocessed_image,
                preprocessed_mask,
                '_' + preprocessing_function.__name__))
    return pd.DataFrame(result).set_index('name')


def cross_validate(method, method_idx, method_name, threshold_values,
                   heatmap, queue):
    images = deepcopy(all_images)
    images_latex = [image.replace('_', '-').replace('OK', '')
                    .replace('ann', '')[:-4]
                    for image in images]
    result = {}
    all_parameters = {}
    # 'False' regression.
    printnow('Cross-validating constant threshold...')
    cross_validated_score = 0
    for i in range(len(images)):
        reduced_heatmap = np.hstack((heatmap[:, :i], heatmap[:, i + 1:]))
        best_line = np.argmax(reduced_heatmap.mean(axis=1))
        cross_validated_score += heatmap[best_line, i]
    cross_validated_score /= len(images)
    result['cross_validated_score_constant'] = cross_validated_score
    scores = heatmap.mean(axis=1)
    best_line = np.argmax(scores)
    result['pseudo_cross_validated_score_constant'] = scores[best_line]
    all_parameters[
        'cross_validated_score_constant'] = threshold_values[best_line]
    if args.plot:
        x, y = np.meshgrid(np.arange(0, len(images) + 1), threshold_values)
        fig = plt.figure()
        plt.pcolormesh(x, y, heatmap, rasterized=True)  # , cmap='Reds')
        # TODO: more method info here
        # plt.title('F1 score by threshold for images in the calibration set\n'
        #           'Preprocessing: ' + method[0].replace('_', ' ') + '\n' +
        #           'Data binding: ' + method[1].replace('_', ' '))
        plt.ylabel("Seuil SPADE")
        plt.plot([0, len(images)], [threshold_values[best_line],
                                    threshold_values[best_line]],
                 label='Seuil SPADE optimal',
                 color='Green')
        if method_idx == 113:
            plt.xticks(np.arange(0.5, len(images)), images_latex, rotation=90)
            plt.xlabel("Nom de l'image")
        else:
            plt.xticks(np.arange(0.5, len(images)), [''] * 102, rotation=90)
        color_bar = plt.colorbar()
        color_bar.set_label("Score $F_1$")
        plt.tick_params(top='off', bottom='off')
        plt.xlim(xmin=0, xmax=len(images_latex))
        plt.yticks(list(plt.yticks()[0]) + [threshold_values[best_line]])
        plt.ylim(ymin=threshold_values[0], ymax=threshold_values[-1])
        plt.legend()
        plt.tight_layout()
        if method_idx == 113:
            fig.set_size_inches(15, 9)
        else:
            fig.set_size_inches(15, 6)
        plt.savefig(os.path.join(output_folder,
                                 '{}.pdf'.format(method_idx)),
                    dpi=300)
        plt.close('all')
    if args.constant_only:
        queue.put((method, result, all_parameters))
        with open(os.path.join(output_folder,
                               '{}.csv'.format(method_idx)), 'w') as fh:
            fh.write('{},{},{}'.format(cross_validated_score,
                                       scores[best_line],
                                       threshold_values[best_line]))
        printnow('Done for ', method_name)
        return
    # 'Real' regression
    printnow('Cross-validating image feature dependant threshold...')
    # Prepare data for regression
    best_scores = heatmap.max(axis=0)
    # image_filter = best_scores >= args.minimal_score
    # images = images[image_filter]
    # heatmap = heatmap[:, image_filter]
    best_scores_threshold_range = np.zeros(len(images), float)
    best_scores_centers = np.zeros(len(images), int)
    for i in range(len(images)):
        where = np.argwhere(heatmap[:, i] == best_scores[i])
        best_scores_threshold_range[i] = \
            threshold_values[where[-1]] - \
            threshold_values[where[0]]
        best_scores_centers[i] = round(np.mean(where))
    for feature in images_statistics:
        feature_values = images_statistics[feature]  # [image_filter]
        feature_values_order = np.argsort(feature_values)
        sorted_best_scores_centers = \
            best_scores_centers[feature_values_order]
        sorted_feature_values = feature_values[feature_values_order]
        sorted_heatmap = heatmap[:, feature_values_order]
        sorted_best_scores_threshold_range = best_scores_threshold_range[
            feature_values_order]
        sorted_best_scores = best_scores[feature_values_order]
        best_regression_score = 0
        for best_score_exponent in best_score_exponents:
            uncertainties = (sorted_best_scores_threshold_range + 1) / \
                            (sorted_best_scores ** best_score_exponent)
            for p0, regression_function in regression_functions.items():
                try:
                    function_name = regression_function.__name__
                    # printnow(feature, ', ', function_name, '...')
                    cross_validated_score = 0
                    # Real cross-validation...
                    for i in range(len(images)):
                        reduced_x = np.hstack((
                            sorted_feature_values[:i],
                            sorted_feature_values[i + 1:]))
                        reduced_y = np.hstack((
                            sorted_best_scores_centers[:i],
                            sorted_best_scores_centers[i + 1:]))
                        reduced_uncertainties = np.hstack((
                            uncertainties[:i],
                            uncertainties[i + 1:]))
                        parameters, covariance = curve_fit(
                            regression_function,
                            reduced_x,
                            threshold_values[reduced_y],
                            p0=p0,  # TODO: find a generic way to pass p0
                            sigma=reduced_uncertainties)
                        predicted_threshold_value = regression_function(
                            feature_values[i], *parameters)
                        cross_validated_score += sorted_heatmap[
                            find_nearest_sorted_idx(
                                threshold_values, predicted_threshold_value),
                            i]
                    cross_validated_score /= len(images)
                    # Pseudo cross-validation
                    parameters, covariance = curve_fit(
                        regression_function,
                        sorted_feature_values,
                        threshold_values[sorted_best_scores_centers],
                        p0=p0,  # TODO: find a generic way to pass p0
                        sigma=uncertainties)
                    predicted_threshold_values = regression_function(
                        sorted_feature_values, *parameters)
                    pseudo_cross_validated_score = 0
                    all_parameters['cross_validated_score_' + feature + '_' +
                                   function_name + '_' +
                                   str(best_score_exponent)] = parameters
                    for i, feature_value in enumerate(feature_values):
                        pseudo_cross_validated_score += sorted_heatmap[
                            find_nearest_sorted_idx(
                                threshold_values,
                                predicted_threshold_values[i]), i]
                    pseudo_cross_validated_score /= len(images)
                    result['pseudo_cross_validated_score_' + feature + '_' +
                           function_name + '_' + str(best_score_exponent)] = \
                        pseudo_cross_validated_score
                    result['cross_validated_score_' + feature + '_' +
                           function_name + '_' + str(best_score_exponent)] = \
                        cross_validated_score
                    if cross_validated_score > best_regression_score:
                        best_regression_function = regression_function.__name__
                        best_regression_score = cross_validated_score
                        best_parameters = parameters
                        best_exponent = best_score_exponent
                        best_predicted_values = predicted_threshold_values
                        best_pseudo_score = pseudo_cross_validated_score
                except RuntimeError:  # do not stop everything if no fit!
                    pass
        if args.plot:
            # Without this pseudo max, image on the right is invisible.
            pseudo_feature_max = \
                sorted_feature_values[-1] + (sorted_feature_values[-1] -
                                             sorted_feature_values[0]) / 10
            x, y = np.meshgrid(np.hstack((sorted_feature_values,
                                          pseudo_feature_max)),
                               threshold_values)
            plt.figure(figsize=(15, 10))
            plt.title(
                'F1 score by threshold by image ' + feature + '\n' +
                'Preprocessing: ' + method[0].replace('_', ' ') + '\n' +
                'Data binding: ' + method[1].replace('_', ' ') + '\n')
            plt.ylabel("SPADE threshold")
            plt.xlabel("Image " + feature)
            plt.pcolormesh(x, y, sorted_heatmap)  # , cmap='Reds')
            plt.plot(sorted_feature_values,
                     best_predicted_values,
                     color='green',
                     label='Regression function:{}\n'
                           'Parameters: {}\n'
                           'Score uncertainty importance: {}\n'
                           'Cross-validated score: {:.2f} ({:.2f})'
                     .format(
                         best_regression_function,
                         best_parameters,
                         best_exponent,
                         best_regression_score,
                         best_pseudo_score
                     ))
            color_bar = plt.colorbar()
            color_bar.set_label("F1 Score")
            plt.plot(sorted_feature_values,
                     threshold_values[sorted_best_scores_centers],
                     marker='*',
                     color='white',
                     alpha=0.8,
                     linestyle='None')
            plt.errorbar(sorted_feature_values,
                         threshold_values[sorted_best_scores_centers],
                         sorted_best_scores_threshold_range,
                         color='white',
                         alpha=0.8,
                         linestyle='None')
            plt.xlim(xmin=sorted_feature_values[0],
                     xmax=pseudo_feature_max)
            plt.ylim(ymin=threshold_values[0], ymax=threshold_values[-1])
            plt.tight_layout()
            plt.legend()
            plt.savefig(os.path.join(
                output_folder,
                '{}_{}_{}.png'.format(method_idx, method_name, feature)))
            plt.xscale('log')
            plt.savefig(os.path.join(
                output_folder,
                '{}_{}_{}_logscale.png'.format(method_idx,
                                               method_name,
                                               feature)))
            plt.close('all')
    printnow('Putting results in queue...')
    queue.put((method, result, all_parameters))
    printnow('Done for ', method_name)


# TODO: use command-line arguments here
results_folder = '/home/nicoco/Documents/INRIA/calibration_results_fixed/'
expertise_folder = '/home/nicoco/Documents/INRIA/calibration_data/expertise/'
output_folder = '/home/nicoco/Documents/INRIA/calibration_summary/'
os.makedirs(output_folder)
best_score_exponents = [0, 1] + np.arange(5, 100, 5).tolist() + [10 ** 4,
                                                                 10 ** 6] + \
                       [float('Inf')]
regression_functions = {(0, 1): log_function,
                        (0, 1): sqrt_function,
                        (50, 1): inverse_function,
                        (0,): polynomial_function,
                        (0, 1): polynomial_function,
                        (0, 1, 1): polynomial_function,
                        (0, 1, 1, 1): polynomial_function,
                        (0, 1, 1, 1, 1): polynomial_function,
                        (0, 1, 1, 1, 1, 1): polynomial_function}
preprocessing_functions = [
    normalize_variance,
    median_filter,
    normalize_variance_median_filter,
    maximum_intensity_projection,
    maximum_intensity_projection_normalize_variance,
    maximum_intensity_projection_normalize_variance_median_filter,
    normalize_by_median,
    normalize_mean]
method_fields = ['preprocessing', 'data_binding', 'centile',
                 'library', 'minsurf']


plt.style.use('ggplot')
plt.rcParams['lines.linewidth'] = 2
plt.rc('font', **{'family': 'serif'})
plt.rcParams['text.latex.unicode'] = True
plt.rc('text', usetex=True)

###############################################################################
# TODO: cleanup and documentation on CLI arguments parser
args = ArgumentParser()
args.add_argument('--cross-validation', dest='cross_validation',
                  action='store_true')
args.add_argument('--combine', dest='combine',
                  action='store_true')
args.add_argument('--debug', dest='debug',
                  action='store_true')
args.add_argument('--constant-only', dest='constant_only',
                  action='store_true')
args.add_argument('--methods-to-cross-validate',
                  dest='methods_to_cross_validate',
                  type=int)
args.add_argument('--minimal-score',
                  dest='minimal_score',
                  type=float)
args.add_argument('--best-ideal-methods',
                  dest='best_ideal_methods',
                  action='store_true')
args.add_argument('--plot',
                  dest='plot',
                  action='store_true')
args.set_defaults(methods_to_cross_validate=float("Inf"),
                  constant_only=False,
                  minimal_score=0,
                  plot=False,
                  debug=False,
                  best_ideal_methods=False,
                  cross_validation=False,
                  combine=False)
args = args.parse_args()
###############################################################################
parameters_test_file_csv = os.path.join(output_folder,
                                        'parameters_test.csv')
parameters_test_file = os.path.join(output_folder,
                                    'parameters_test.pickle')
if args.combine:
    printnow('Combining results...')
    csv_files = os.listdir(results_folder)
    if args.debug:
        csv_files = csv_files[:15]
    with open(parameters_test_file_csv, 'w') as output_fh:
        for i, csvfile in enumerate(csv_files):
            printnow("Loading file {} ({}/{})...".format(csvfile, i + 1,
                                                         len(csv_files)))
            with open(os.path.join(results_folder, csvfile)) as input_fh:
                image_name = csvfile.split('.tif_')[0] + '.tif'
                if i == 0:
                    output_fh.write('image,' + next(input_fh))
                else:
                    # TODO: find the clean way to do this
                    for line in input_fh:
                        break
                for line in input_fh:
                    output_fh.write(image_name + ',' + line)
    printnow("Loading big file...")
    parameters_test = pd.read_csv(parameters_test_file_csv, index_col=False)
    exp = read_expertises(expertise_folder)
    expn = {i: len(exp[i]) for i in exp}
    printnow("Adding expertises and scores...")
    add_scores_and_groundtruths(parameters_test, expn, gt='ex')
    parameters_test.fillna(0, inplace=True)
    printnow("Saving big file...")
    parameters_test.to_pickle(parameters_test_file)
###############################################################################
if args.best_ideal_methods:
    try:
        printnow("Loading big file...")
        parameters_test = pd.read_pickle(parameters_test_file)
    except FileNotFoundError:
        printnow('Cannot find parameters_test.csv\n'
                 'Please use --combine first.')
        sys.exit(1)
    printnow("Determining best 'ideal' images...")
    best_images_scores = best_multiple(parameters_test, 'image', 'score')
    best_images_scores = best_unique(best_images_scores,
                                     method_fields + ['image'],
                                     'score').sort_values(by='image')
    best_images_scores.to_csv(os.path.join(output_folder,
                                           'ideal_images.csv'))
    printnow("Determining best 'ideal' methods...")
    best_images_scores_by_method = best_unique(parameters_test,
                                               ['image'] + method_fields,
                                               'score')
    best_methods = best_images_scores_by_method.groupby(method_fields) \
        .agg({'tp': np.sum,
              'fp': np.sum,
              'fn': np.sum,
              'dd': np.sum,
              'ex': np.sum,
              'image': pd.Series.nunique,
              'score': pd.Series.mean})
    best_methods.rename(columns={'score': 'score_ideal_image_wise'},
                        inplace=True)
    add_scores(best_methods, gt='ex')
    best_methods.rename(columns={'score': 'score_ideal_particle_wise'},
                        inplace=True)

    best_methods = best_methods.fillna(0).sort_values(
            by='score_ideal_particle_wise')[::-1]
    # printnow('Counting number of tests for each method...')
    # for method in parameters_test.index:
    #     best_methods.loc[method, 'number_of_tested_thresholds'] = \
    #         len(parameters_test.loc[method].index)
    best_methods.to_csv(os.path.join(output_folder, 'best_ideal_methods.csv'))
else:
    printnow("Loading best 'ideal' methods...")
    try:
        best_methods = pd.read_csv(
            os.path.join(output_folder, 'best_ideal_methods.csv'),
            index_col=False).set_index(method_fields)
    except FileNotFoundError:
        printnow('Cannot find best_ideal_methods.csv\n'
                 'Please use --best-ideal-methods first.')
        sys.exit(1)
###############################################################################
if args.cross_validation:
    pickle_file = os.path.join(output_folder, 'parameter_tests_indexed.pickle')
    if exists(pickle_file) and not args.combine:
        printnow('Loading the big indexed data frame...')
        parameters_test = pd.read_pickle(pickle_file)
    else:
        printnow('Indexing the big data frame...')
        try:
            parameters_test.iloc[0]
        except:
            parameters_test = pd.read_pickle(parameters_test_file)
        parameters_test.set_index(method_fields + ['image', 'threshold'],
                                  inplace=True)
        parameters_test.sort_index(inplace=True)
        printnow('Saving the the big indexed data frame...')
        parameters_test.to_pickle(pickle_file)
    all_images = os.listdir('../../calibration_data/images')
    if args.debug:
        all_images = all_images[:10]
        preprocessing_functions = []
        best_score_exponents = [5, 10]
        args.methods_to_cross_validate = 4
    if not args.constant_only:
        printnow('Getting images characteristics...')
        images_statistics = get_images_statistics(all_images)
    found_methods = []
    printnow('Choosing up to {} methods to cross-validate...'.format(
        args.methods_to_cross_validate))
    queues = []
    processes = []
    evaluated_methods = 0
    for method_idx in range(len(best_methods.index)):
        if exists(os.path.join(output_folder,
                               '{}.csv'.format(method_idx))):
            printnow('Already done.')
            continue
        method = best_methods.iloc[method_idx].name
        try:
            if parameters_test.loc[method].index.get_level_values(
                    'image').nunique() != len(all_images):
                printnow('Data missing for {}, skipping!'.format(method))
                continue
        except KeyError:
            printnow('Problem with {}, skipping!'.format(method))
            continue
        if 'normalize' in method[0] and  best_methods.iloc[
                method_idx].score_ideal_image_wise > args.minimal_score:
            printnow(method, '\n',
                     best_methods.iloc[method_idx].score_ideal_particle_wise)
            found_methods += [method[:2]]
            method_name = "_".join(method[:2])
            threshold_values = parameters_test.loc[
                method].index.get_level_values(
                'threshold').sort_values().unique()
            heatmap = np.zeros((len(threshold_values), len(all_images)))
            printnow('Filling heatmap...')
            for j, image in enumerate(all_images):
                # printnow(j, ' ', image)
                threshold_values_for_image = parameters_test.loc[
                    method + (image,)].index.values
                scores_values_for_image = parameters_test.loc[
                    method + (image,), 'score'].values
                for i, threshold in enumerate(threshold_values):
                    closest = find_nearest_sorted_idx(
                        threshold_values_for_image,
                        threshold)
                    # If outside threshold range, leave 0 as score.
                    if closest != 0 and closest != len(
                            threshold_values_for_image) - 1:
                        heatmap[i, j] = scores_values_for_image[closest]
            queues += [Queue()]
            processes += [Process(target=cross_validate,
                                  args=(method,
                                        method_idx,
                                        method_name,
                                        threshold_values,
                                        heatmap,
                                        queues[-1]))]
            processes[-1].start()
            evaluated_methods += 1
            if evaluated_methods >= args.methods_to_cross_validate:
                break
    printnow('Waiting for all processes to end and combining results...')
    names = []
    scores = []
    parameters = []
    for queue in queues:
        result = queue.get()
        names += [result[0]]
        scores += [result[1]]
        parameters += [result[2]]
    maximum_score = 0
    best_criteria = None
    best_method = None
    calibrated_methods = []
    printnow('Determining real best method...')
    for i, score in enumerate(scores):
        best_score_for_method = 0
        method_dict = {}
        for field, value in zip(best_methods.index.names, names[i]):
            method_dict[field] = value
        for score_name in score:
            if score[score_name] > best_score_for_method and not \
                    score_name.startswith('pseudo'):
                best_score_for_method = score[score_name]
                method_dict['pseudo cross validated score'] = score[
                    'pseudo_' + score_name]
                method_dict['cross validated score'] = score[score_name]
                method_dict['calibration method'] = score_name[22:]
                method_dict['parameters'] = parameters[i][score_name]
                method_dict['ideal score image wise'] = best_methods.loc[
                    names[i], 'score_ideal_image_wise']
                method_dict['ideal score particle wise'] = best_methods.loc[
                    names[i], 'score_ideal_particle_wise']
                if score[score_name] > maximum_score and not \
                        score_name.startswith('pseudo'):
                    maximum_score = score[score_name]
                    best_criteria = score_name
                    best_method = names[i]
                    best_parameters = parameters[i][score_name]
        calibrated_methods += [method_dict]
    calibrated_methods = pd.DataFrame(calibrated_methods) \
                             .sort_values(by='cross validated score')[::-1]
    printnow('\nWinning method: {}\n'
             'Calibration: {}\n'
             'Parameters: {}\n'
             'Score: {}'.format(best_method,
                                best_criteria,
                                best_parameters,
                                maximum_score))
    calibrated_methods[method_fields + ['cross validated score',
                                        'pseudo cross validated score',
                                        'calibration method',
                                        'parameters',
                                        'ideal score image wise',
                                        'ideal score particle wise']] \
        .to_csv(os.path.join(output_folder,
                             'best_calibrated_methods.csv'))
    ideal_images_with_best_calibrated_method = best_unique(
        parameters_test.loc[best_method].reset_index(), 'image', 'score')
    ideal_images_with_best_calibrated_method.to_csv(os.path.join(
        output_folder, 'ideal_images_with_best_calibrated_method.csv'))
###############################################################################
printnow('Done.')
