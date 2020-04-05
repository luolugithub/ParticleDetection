"""
This module provides the core of SPADE, namely the spade2d function.
"""

import numpy as np


class SpadeCandidates:
    """
    Candidates list for SPADE detected objects.
    """

    def __init__(self, max_candidates, shapes, image_shape):
        self.list = np.zeros((max_candidates, 3), np.uint32)
        self.scores = np.zeros(max_candidates, np.float64)
        self.idx = np.array(0, np.int64)
        self.y, self.x = image_shape
        self.shapes = shapes
        self.labeled_image = np.zeros(
            (self.y + shapes.image_extension_size * 2,
             self.x + shapes.image_extension_size * 2),
            dtype=np.uint16)

    def get_list(self):
        return self.list[:self.idx]

    def get_scores(self):
        return self.scores[:self.idx]

    # TODO: handle RuntimeWarnings raised here
    def add(self, scores, threshold, x, y, possible_shapes_idx):
        valid_scores = np.where(scores >= threshold)
        n_candidates = len(valid_scores[0])
        if n_candidates:
            self.list[self.idx:self.idx + n_candidates, 0] = y
            self.list[self.idx:self.idx + n_candidates, 1] = x
            self.list[self.idx:self.idx + n_candidates, 2] = \
                possible_shapes_idx[0][valid_scores]
            self.scores[self.idx:self.idx + n_candidates] = \
                scores[valid_scores].ravel()
            self.idx += n_candidates

    def sort(self):
        self.list = \
            self.list[:self.idx][self.scores[:self.idx].argsort()][::-1]

    def eliminate_overlapping(self, object_idx_start=1, update_list=False,
                              max_objects=float('Inf')):
        i = 0
        ext_size = self.shapes.image_extension_size
        one = np.array(1, np.uint16)
        # obj number to fill label matrix with
        object_idx = np.array(object_idx_start, np.uint16)
        number_of_candidates = len(self.list)
        object_indices = np.zeros(number_of_candidates, dtype=np.uint32)
        while i < number_of_candidates:
            y, x, shape_id = self.list[i, :]
            local_labeled_image = self.labeled_image[
                                  y:y + self.shapes.grid_size,
                                  x:x + self.shapes.grid_size]
            # if no overlap...
            if not local_labeled_image[self.shapes[shape_id]].any():
                local_labeled_image += self.shapes[shape_id] * object_idx
                object_indices[object_idx - object_idx_start] = i
                object_idx += one
                if object_idx >= max_objects:
                    break
            i += 1
        self.labeled_image = self.labeled_image[ext_size:-ext_size,
                        ext_size:-ext_size]
        if update_list:
            # If return_cand=True, add a last column indicating object number
            # in the labeled image.
            self.list = np.hstack((
                self.list[object_indices[:object_idx - object_idx_start]] -
                    [ext_size, ext_size, 0],
                np.arange(object_idx_start, object_idx)[:, np.newaxis]))


# TODO: write a long and detailed docstring here
def spade2d(image, shapes_library, threshold, data_binding_function=None,
            mask=None, potential_centers=None, return_candidates_list=False,
            object_idx_start=1, max_candidates=10000000):
    """
    Main SPADE function.
    """
    # If no mask or potential centers are passed, creates bool arrays filled
    # with True, same dimensions as image, i.e. inspect all pixels.
    if mask is None:
        mask = np.ones_like(image, bool)
    if potential_centers is None:
        potential_centers = mask
    # Preallocate memory for potential candidates list.
    candidates = SpadeCandidates(max_candidates,
                                 shapes_library,
                                 image.shape)
    # We need to extend the image because it is possible for a object 
    # coordinates to be negative because the shapes coordinates is the top
    # left corner.
    extended_image = np.pad(image.astype(np.float64),
                            shapes_library.image_extension_size,
                            'constant')
    extended_mask = np.pad(mask, shapes_library.image_extension_size,
                           'constant')
    inverted_mask = np.logical_not(extended_mask)
    extended_potential_centers = np.pad(potential_centers,
                                        shapes_library.image_extension_size,
                                        'constant')
    for centered_y, centered_x in np.argwhere(extended_potential_centers):
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
            possible_shapes = shapes_library[possible_shapes_idx]
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
            if data_binding_function is None:
                # If no data binding function is passed to the detector,
                # use the faster equivalent of mean_difference.
                rings_means = dot_means(masked_rings,
                                        extended_image[ring_window])
                shapes_means = dot_means(possible_shapes,
                                         extended_image[shape_window])
                scores = shapes_means - rings_means
            else:
                shapes_pixels = np.where(shapes_library[possible_shapes_idx],
                                         extended_image[shape_window],
                                         np.nan)
                rings_pixels = np.where(masked_rings,
                                        extended_image[ring_window],
                                        np.nan)
                scores = data_binding_function(shapes_pixels, rings_pixels)
            candidates.add(scores, threshold, real_x, real_y,
                           possible_shapes_idx)
    # Sort candidates by scores.
    candidates.sort()
    candidates.eliminate_overlapping(
        object_idx_start=object_idx_start,
        update_list=True if return_candidates_list else False)

    if return_candidates_list:
        return candidates.labeled_image, candidates.list
    else:
        return candidates.labeled_image


def dot_means(mask, array, ax=(1, 2)):
    # Fast way to return an array of masked (by another array) means.
    # Thanks Divakar from stack overflow for the idea.
    counts = mask.sum(axis=ax)
    # TODO: handle RuntimeWarnings raised here
    return np.dot(mask.reshape(mask.shape[0], -1), array.ravel()) / counts
