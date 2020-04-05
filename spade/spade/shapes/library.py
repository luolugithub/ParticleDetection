"""
This file provides the ShapesLibrary class needed for SPADE.
"""

from os.path import dirname, join
import warnings

import numpy as np
from scipy.ndimage.morphology import binary_dilation


class ShapesLibrary(np.ndarray):
    """Shapes library class for SPADE"""

    # TODO: maybe make size restriction easier?
    def __new__(cls, input_array, name="Unnamed",
                ring_distance=0, ring_thickness=1,
                sort=True):
        if sort:
            input_array = np.array(sorted(input_array, key=np.sum), bool)
        elif type(input_array) != np.ndarray:
            input_array = np.asarray(input_array, bool)

        if input_array.ndim != 3 or input_array.shape[1] != \
                input_array.shape[2]:
            raise TypeError('Input array must be in the shape '
                            'number of shapes x grid_size x grid_size')

        obj = input_array.view(cls)
        obj.name = name
        obj.surfaces = get_surfaces(input_array)
        obj.rings = get_rings(input_array,
                              ring_distance=ring_distance,
                              ring_thickness=ring_thickness)
        obj.grid_size = input_array.shape[-1]
        obj.shape_start = start_dict(obj.surfaces)

        obj.ring_distance = ring_distance
        obj.ring_thickness = ring_thickness

        obj.half_shape_size = obj.grid_size // 2
        obj.image_extension_size = \
            obj.half_shape_size + ring_distance + ring_thickness
        obj.ring_window_padding_size = ring_distance + ring_thickness
        obj.ring_window_limit = obj.grid_size + obj.ring_window_padding_size

        if not input_array[:, obj.half_shape_size, obj.half_shape_size].all():
            warnings.warn("At least one shape doesn't include the central "
                          "grid pixel. This may cause weird behaviour.")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.surfaces = getattr(obj, 'surfaces', None)
        self.rings = getattr(obj, 'rings', None)
        self.grid_size = getattr(obj, 'grid_size', None)
        self.shape_start = getattr(obj, 'shape_start', None)
        self.image_extension_size = getattr(obj, 'image_extension_size', None)
        self.half_shape_size = getattr(obj, 'half_shape_size', None)
        self.ring_window_padding_size = getattr(obj,
                                                'ring_window_padding_size',
                                                None)
        self.ring_window_limit = getattr(obj, 'ring_window_limit', None)
        self.ring_distance = getattr(obj, 'ring_distance', None)
        self.ring_thickness = getattr(obj, 'ring_thickness', None)

        # def update(self):
        #     shapes_array = np.asarray(self)
        #     self.surfaces = get_surfaces(shapes_array)
        #     self.rings = get_rings(shapes_array)
        #     self.shape_start = start_dict(self.surfaces)


def get_surfaces(array):
    return array.sum(axis=tuple(np.arange(1, array.ndim)))


def start_dict(array):
    return {k: np.searchsorted(array, k) for k in np.unique(array)}


def get_rings(array, ring_distance, ring_thickness):
    big_dilation_size = 1 + (ring_distance + ring_thickness) * 2
    small_dilation_size = big_dilation_size - ring_thickness * 2
    big_structuring_element = np.ones((3,
                                       big_dilation_size,
                                       big_dilation_size), bool)
    small_structuring_element = np.ones((3,
                                         small_dilation_size,
                                         small_dilation_size), bool)
    big_structuring_element[[0, -1]] = 0
    small_structuring_element[[0, -1]] = 0
    pad_size = ring_distance + ring_thickness
    padded_array = np.pad(array,
                          pad_size,
                          'constant')[pad_size:-pad_size]
    big_dilation = binary_dilation(padded_array, big_structuring_element)
    small_dilation = binary_dilation(padded_array, small_structuring_element)
    rings = np.logical_xor(big_dilation, small_dilation)
    # print(rings[-1:].astype(int))
    # a = input()
    return rings