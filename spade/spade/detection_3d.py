"""
This module provides functions to detect particles using SPADE on each slice
of a 3D image stack, and then combining them into 3D particles.
"""

import numpy as np
from .detection_2d import spade2d


def spade3d(image3d, shapes_library, threshold,
            mask=None,
            potential_centers=None,
            data_binding_function=None,
            minimal_surface_3ds=1,
            minimal_z_thickness=1,
            minimal_z_overlap=0.25,
            max_candidates=10000):
    # TODO: docstring
    candidates = np.zeros((max_candidates, 5), np.int16)
    candidates_idx = 0
    labeled_image_3d = np.zeros_like(image3d)
    if mask is None:
        mask = [None] * image3d.shape[0]
    if potential_centers is None:
        potential_centers = mask
    for z, image2d in enumerate(image3d):
        labeled_image_3d[z, :, :], tmp_candidates = \
            spade2d(image2d,
                    shapes_library,
                    threshold,
                    data_binding_function=data_binding_function,
                    mask=mask[z],
                    potential_centers=potential_centers[z],
                    return_candidates_list=True,
                    object_idx_start=candidates_idx + 1)
        candidates[candidates_idx: candidates_idx +
                                   len(tmp_candidates), 1:] = tmp_candidates
        candidates[candidates_idx: candidates_idx + len(tmp_candidates), 0] = z
        candidates_idx += len(tmp_candidates)
    candidates = candidates[:candidates_idx]
    return combine_2d_shapes(labeled_image_3d, candidates,
                             shapes_library, minimal_surface_3ds,
                             minimal_z_thickness, minimal_z_overlap)


def combine_2d_shapes(labmat, candidates, shapes, minsurf, minz, ztol):
    # now candidates are [z, y, x, shapeid, objectid], sorted by z, objectid is
    # the id in the labelmatrix
    finallabelmatrix = np.zeros(labmat.shape, dtype=np.uint8)
    zmax = labmat.shape[0]
    nbcand = len(candidates)
    i = 0
    objid = 1
    exclude = []
    potobj = np.zeros(zmax, dtype=np.uint16)
    while i < nbcand:
        if candidates[i, 4] not in exclude:
            z1, y1, x1, shapeid1, oid1 = candidates[i, :]
            potobjlen = 1
            potobj[0] = oid1
            sameobject = True
            targz = z1 + 1
            j = i + 1
            while sameobject:
                sameobject = False
                while j < nbcand:
                    if candidates[j][0] == z1:
                        j += 1
                    elif candidates[j][0] > targz:
                        break
                    else:
                        z2, y2, x2, shapeid2, oid2 = candidates[j, :]
                        if overlap(labmat, z1, z2, oid1, oid2) / \
                                shapes.surfaces[min(shapeid1, shapeid2)] >= \
                                ztol:
                            potobj[potobjlen] = oid2
                            potobjlen += 1
                            sameobject = True
                            z1, y1, x1, shapeid1, oid1 = z2, y2, x2, \
                                                         shapeid2, oid2
                            targz = z1 + 1
                            j += 1
                            break
                        j += 1
            if potobjlen >= minz:
                valid = False
                for sid in candidates[potobj[:potobjlen] - 1, 3]:
                    if shapes.surfaces[sid] >= minsurf:
                        valid = True
                        break
                if valid:
                    rpotobj = potobj[:potobjlen].tolist()
                    exclude += rpotobj
                    splitted_obj = np.logical_or.reduce([labmat == x for x
                                                         in rpotobj])
                    finallabelmatrix[splitted_obj] = objid
                    objid += 1
        i += 1
    return finallabelmatrix


def overlap(labmat, z1, z2, oid1, oid2):
    loclabmat = labmat[z1:z2 + 1, :, :]
    loclabmat = np.where(np.logical_or(loclabmat == oid1,
                                       loclabmat == oid2), True, False)
    return np.logical_and.reduce(loclabmat, 0).sum()
