# TODO: one function = one preprocess (easier said than done), i.e. no
# redundancy

import numpy as np
from skimage.filters.rank import median
from skimage.morphology import disk, square
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing


def normalize_variance(image, mask):
    return image / np.var(image[mask]), np.copy(mask)


def do_nothing(image, mask):
    return np.copy(image), np.copy(mask)


def median_filter(image, mask):
    new_image = np.copy(image)
    for z, z_slice in enumerate(image):
        new_image[z] = median(z_slice, square(3))
    return new_image, np.copy(mask)


def normalize_variance_median_filter(image, mask):
    normalized_image, new_mask = normalize_variance(image, mask)
    for z, z_slice in enumerate(normalized_image):
        normalized_image[z] = median(z_slice, square(3))
    return normalized_image, new_mask


def maximum_intensity_projection(image, mask):
    mip = np.array([np.max(image, axis=0)])
    new_mask = mip > threshold_otsu(mip)
    new_mask[0] = binary_closing(new_mask[0], disk(3))
    return mip, new_mask


def maximum_intensity_projection_normalize_variance(image, mask):
    mip = np.array([np.max(image, axis=0)])
    new_mask = mip > threshold_otsu(mip)
    new_mask[0] = binary_closing(new_mask[0], disk(3))
    return mip / np.var(mip[new_mask]), new_mask


def maximum_intensity_projection_normalize_variance_median_filter(image, mask):
    mip = np.array([np.max(image, axis=0)])
    mip[0] = median(mip[0], square(3))
    new_mask = mip > threshold_otsu(mip)
    new_mask[0] = binary_closing(new_mask[0], disk(3))
    return mip / np.var(mip[new_mask]), new_mask


def normalize_by_median(image, mask):
    new_image = image / np.median(image[mask])
    return new_image, np.copy(mask)


def maximum_intensity_projection_normalize_by_median(image, mask):
    mip = np.array([np.max(image, axis=0)])
    new_mask = mip > threshold_otsu(mip)
    new_mask[0] = binary_closing(new_mask[0], disk(3))
    return mip / np.median(mip[new_mask]), new_mask


def normalize_mean(image, mask):
    new_image = image / np.mean(image[mask])
    return new_image, np.copy(mask)


def scale_reduce(image, mask):
    new_image = image / np.std(image[mask])
    new_image = new_image - new_image.mean()
    return new_image, np.copy(mask)


def maximum_intensity_projection_normalize_mean(image, mask):
    mip = np.array([np.max(image, axis=0)])
    new_mask = mip > threshold_otsu(mip)
    new_mask[0] = binary_closing(new_mask[0], disk(3))
    return mip / np.mean(mip[new_mask]), new_mask


def normalize_mean_maximum_intensity_projection(image, mask):
    new_image = image / np.mean(image[mask])
    mip = np.array([np.max(new_image, axis=0)])
    new_mask = mip > threshold_otsu(mip)
    new_mask[0] = binary_closing(new_mask[0], disk(3))
    return mip, new_mask
