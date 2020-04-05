import numpy as np
from spade.shapes.library import ShapesLibrary


# from spade.detection_2d import eliminate_overlapping_candidates


class SpadeScores:
    def __init__(self, image, mask, shapes_library, data_binding_function):
        if image.ndim != 2:
            raise TypeError('Image must be 2-Dimensional')

        self.data_binding_function_name = data_binding_function.__name__
        self.shapes_library = shapes_library

        extended_image = np.pad(image, shapes_library.image_extension_size,
                                'constant')

        self.array = np.empty((extended_image.shape +
                               (len(shapes_library),)), np.float64)
        self.array.fill(np.nan)

        extended_mask = np.pad(mask, shapes_library.image_extension_size,
                               'constant')

        inverted_mask = np.logical_not(extended_mask)

        for centered_y, centered_x in np.argwhere(extended_mask):
            real_y = centered_y - shapes_library.half_shape_size
            real_x = centered_x - shapes_library.half_shape_size
            shape_window = (slice(real_y,
                                  real_y + shapes_library.grid_size),
                            slice(real_x,
                                  real_x + shapes_library.grid_size)
                            )
            local_inverted_mask = inverted_mask[shape_window]
            possible_shapes_bool = np.logical_not(
                (shapes_library & local_inverted_mask).any(axis=(1, 2)))
            if possible_shapes_bool.any():
                possible_shapes_idx = np.where(possible_shapes_bool)
                shapes_pixels = np.where(shapes_library[possible_shapes_idx],
                                         extended_image[shape_window],
                                         np.nan)

                ring_window = (slice(real_y -
                                     shapes_library.ring_window_padding_size,
                                     real_y + shapes_library.ring_window_limit),
                               slice(real_x -
                                     shapes_library.ring_window_padding_size,
                                     real_x + shapes_library.ring_window_limit)
                               )
                ring_mask = extended_mask[ring_window]
                masked_rings = shapes_library.rings[possible_shapes_idx] & \
                               ring_mask
                rings_pixels = np.where(masked_rings,
                                        extended_image[ring_window],
                                        np.nan)
                self.array[real_y, real_x, possible_shapes_idx] = \
                    data_binding_function(shapes_pixels, rings_pixels)

        self.reduced_array = self.array
        self.reduced_shapes_library = self.shapes_library

    def get_result(self, threshold, return_candidates_list=False,
                   obj_idx_start=1):
        # get indices where scores are above threshold
        # TODO: handle RuntimeWarnings raised here
        candidates_positions = np.where(self.reduced_array >= threshold)
        # transform it into a list of objects
        candidates = np.array(candidates_positions).transpose()
        # sort by score
        candidates = candidates[np.lexsort((candidates[:, 0],
                                            self.reduced_array[
                                                candidates_positions]))]
        # reverse list to have best scores first
        candidates = candidates[::-1]
        # eliminate overlapping shapes
        result = eliminate_overlapping_candidates(
            candidates,
            self.reduced_shapes_library,
            self.reduced_array.shape[:2],
            return_candidates_list,
            obj_idx_start)
        return result

    def get_max_score(self):
        return np.nanmax(self.reduced_array)

    def reduce_array(self, minimal_shape_surface=None, exclude_pixels=None):
        if minimal_shape_surface is None:
            scores_start = 0
        else:
            scores_start = self.shapes_library.shape_start[
                minimal_shape_surface]

        self.reduced_array = self.array[:, :, scores_start:]

        if exclude_pixels is None:
            exclude_pixels = np.zeros_like(self.reduced_array, bool)
        else:
            exclude_pixels = np.roll(
                np.roll(
                    np.pad(exclude_pixels,
                           self.shapes_library.image_extension_size,
                           'constant'),
                    - self.shapes_library.half_shape_size,
                    0),
                - self.shapes_library.half_shape_size, 1)
            exclude_pixels = np.array(
                [exclude_pixels] * self.reduced_array.shape[2]).transpose(1,
                                                                          2,
                                                                          0)

        self.reduced_shapes_library = ShapesLibrary(
            self.shapes_library[scores_start:],
            ring_distance=self.shapes_library.ring_distance,
            ring_thickness=self.shapes_library.ring_thickness
        )
        self.reduced_array[exclude_pixels] = np.nan


def eliminate_overlapping_candidates(candidates, shapes, imshape,
                                     return_cand=False,
                                     object_idx_start=1,
                                     max_objects=float('Inf')):
    """
    This function takes a list of candidates in the form of a numpy array as
    follows [[y, x, shape_id],
             [..., ..., ...]]
    and returns a labeled image, and optionally a list of objects.
    It is possible to specify the starting index in the labeled image (useful
    for spade_3d)
    """
    y, x = imshape
    i = 0
    ext_size = shapes.image_extension_size
    labeled_image = np.zeros((y, x), dtype=np.uint16)
    one = np.array(1, np.uint16)
    # obj number to fill label matrix with
    object_idx = np.array(object_idx_start, np.uint16)
    number_of_candidates = len(candidates)
    object_indices = np.zeros(number_of_candidates, dtype=np.uint32)
    while i < number_of_candidates:
        y, x, shapeid = candidates[i, :]
        local_labeled_image = labeled_image[y:y + shapes.grid_size,
                              x:x + shapes.grid_size]
        if not local_labeled_image[shapes[shapeid]].any():  # if no overlap
            local_labeled_image += shapes[shapeid] * object_idx
            object_indices[object_idx - object_idx_start] = i
            object_idx += one
            if object_idx >= max_objects:
                break
        i += 1
    labeled_image = labeled_image[ext_size:-ext_size,
                    ext_size:-ext_size]
    if not return_cand:
        return labeled_image
    else:
        # If return_cand=True, add a last column indicating object number in
        # the labeled image.
        candidates = np.hstack((
            candidates[object_indices[:object_idx - object_idx_start]] -
            [ext_size, ext_size, 0],
            np.arange(object_idx_start, object_idx)[:, np.newaxis]))
        return labeled_image, candidates
