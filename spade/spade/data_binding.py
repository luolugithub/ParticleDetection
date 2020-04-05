"""
This module provides different data binding functions to be used with SPADE.
In order to be able to be used with SPADE, a data binding function must:
- take 2 3D numpy arrays containing NaN entries as input
- output a 1D numpy array of spade 'scores' corresponding to the comparison of
the non-NaN values of each of the pixels population along the axis 0.

Example:

Input 1: shapes_pixels = [[[np.nan, np.nan, np.nan],
                           [np.nan,     99, np.nan],
                           [np.nan,     99, np.nan]],
                          [[np.nan,     99, np.nan],
                           [    99,     99, 99],
                           [np.nan,     99, np.nan]]]
Input 2: rings_pixels = [[[np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan,     44,     44,     44, np.nan],
                          [np.nan,     44, np.nan,     44, np.nan],
                          [np.nan,     44, np.nan,     44, np.nan],
                          [np.nan,     44,     44,     44, np.nan]],
                         [[np.nan,     44,     44,     44, np.nan],
                          [    44,     44, np.nan,     44,     44],
                          [    44, np.nan, np.nan, np.nan,     44],
                          [    44,     44, np.nan,     44,     44],
                          [np.nan,     44,     44,     44, np.nan]]]

Output: np.array([55, 55]) if the we expect a mean difference.
"""

import numpy as np
try:
    from scipy.stats import ttest_ind_from_stats as welch_test
except ImportError:
    pass  # TODO: implement a welch_test that doesn't require scipy.stats


def mean_difference(shapes_pixels, rings_pixels):
    return np.nanmean(shapes_pixels, axis=(1, 2)) - np.nanmean(rings_pixels,
                                                               axis=(1, 2))


def median_difference(shapes_pixels, rings_pixels):
    return np.nanmedian(shapes_pixels, axis=(1, 2)) - \
           np.nanmedian(rings_pixels, axis=(1, 2))


def max_min_difference(shapes_pixels, rings_pixels):
    return np.nanmin(shapes_pixels, axis=(1, 2)) - np.nanmax(rings_pixels,
                                                             axis=(1, 2))


# TODO: remove redundant calculations for these 3 functions
def t_test(shapes_pixels, rings_pixels):
    shapes_pixels_means = np.nanmean(shapes_pixels, axis=(1, 2))
    rings_pixels_means = np.nanmean(rings_pixels, axis=(1, 2))
    shapes_pixels_std = np.nanstd(shapes_pixels, axis=(1, 2))
    rings_pixels_std = np.nanstd(rings_pixels, axis=(1, 2))
    shapes_pixels_nobs = (~np.isnan(shapes_pixels)).sum(axis=(1, 2))
    rings_pixels_nobs = (~np.isnan(rings_pixels)).sum(axis=(1, 2))
    result = 1 - welch_test(shapes_pixels_means,
                            shapes_pixels_std,
                            shapes_pixels_nobs,
                            rings_pixels_means,
                            rings_pixels_std,
                            rings_pixels_nobs,
                            equal_var=False)[1]
    result[shapes_pixels_means - rings_pixels_means < 0] = -1
    return result


def bhattacharyya_distance(shapes_pixels, rings_pixels):
    shapes_pixels_variance = np.nanvar(shapes_pixels, axis=(1, 2))
    rings_pixels_variance = np.nanvar(rings_pixels, axis=(1, 2))
    shapes_pixels_variance[shapes_pixels_variance == 0] = 1
    rings_pixels_variance[rings_pixels_variance == 0] = 1
    mean_differences = mean_difference(shapes_pixels, rings_pixels)
    mean_differences[mean_differences <= 0] = np.nan
    result = -np.exp(-(1 / 4 * np.log(1 / 4 * (shapes_pixels_variance /
                                               rings_pixels_variance +
                                               rings_pixels_variance /
                                               shapes_pixels_variance + 2)) +
                       1 / 4 * (mean_differences ** 2) /
                       (shapes_pixels_variance + rings_pixels_variance))) + 1
    return result


def pseudo_bhattacharyya_distance(shapes_pixels, rings_pixels):
    shapes_pixels_variance = np.nanvar(shapes_pixels, axis=(1, 2))
    rings_pixels_variance = np.nanvar(rings_pixels, axis=(1, 2))
    shapes_pixels_variance[shapes_pixels_variance == 0] = 0.1
    rings_pixels_variance[rings_pixels_variance == 0] = 0.1
    mean_differences = mean_difference(shapes_pixels, rings_pixels)
    mean_differences[mean_differences <= 0] = np.nan
    result = -np.exp(-(1 / 4 * np.log(1 / 4 * (shapes_pixels_variance /
                                               rings_pixels_variance +
                                               rings_pixels_variance /
                                               shapes_pixels_variance + 2)) +
                       1 / 4 * (mean_differences ** 2) /
                       (np.sqrt(shapes_pixels_variance +
                                rings_pixels_variance)))) + 1
    return result
