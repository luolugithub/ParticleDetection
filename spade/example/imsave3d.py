from skimage.io import imsave
import numpy as np


def random_bright_colors(threshold=0.5):
    while True:
        rgb = np.array([0, 0, 0])
        while rgb.sum() < threshold:
            rgb = np.random.random(3)
        yield rgb


def imsave3d_overlay(file_name, grayscale_image3d, mask, labeled_image,
                     background_color=(0.1, 0, 0.1),
                     colors=random_bright_colors()):
    maximum_intensity = grayscale_image3d.max()
    rgb_image = np.array([(grayscale_image3d / maximum_intensity)] * 3) \
        .transpose(1, 2, 3, 0).astype(np.float16)
    original_rgb_image = np.copy(rgb_image)
    rgb_image[np.logical_not(mask)] = background_color
    for object_idx in range(1, labeled_image.max() + 1):
        rgb_image[labeled_image == object_idx] = next(colors)
    imsave(file_name, (np.concatenate((rgb_image, original_rgb_image),
                                     axis=2)*255).astype(np.uint8))
