"""
This file was used to generate potatoids_5x5 and potatoids_3x3
You shouldn't need it!
"""
# TODO: still a lot of cleaning

import numpy as np
from scipy import ndimage


# from spade.shapes.library import ShapesLibrary


def intervalstoboolarray(intervals, grid_size):
    # generates an array from a list of intervals
    array = np.zeros((grid_size + 2, grid_size + 2), dtype=np.bool)
    for i, interval in enumerate(intervals):
        if interval:
            start = interval[0]
            end = interval[1]
            array[i + 1] = [False] * (start + 1) + (end - start + 1) * [
                True] + (grid_size - end) * [False]
        else:
            array[i + 1] = [False] * (grid_size + 2)
    return array


def finalarray(shapes, sepcolor, septhickness, outputwidth):
    print('Creating the big array...')
    grid_size = np.shape(shapes[0])[0]
    shapesbyline = outputwidth // (grid_size + 2 + septhickness)
    outputwidth = shapesbyline * (grid_size + 2 + septhickness) - 1
    sep = np.reshape(
        np.array([[sepcolor] * grid_size * septhickness], dtype=np.uint8),
        (grid_size, septhickness))
    seph = np.array([sepcolor] * outputwidth, dtype=np.uint8)
    res = seph
    for i in range(0, len(shapes), shapesbyline):
        rest = shapes[i]
        for forme in shapes[i + 1:i + shapesbyline]:
            rest = np.concatenate([rest, sep, forme], axis=1)
        while rest.shape[1] < len(seph):
            rest = np.concatenate([rest, sep], axis=1)
        res = np.vstack([res, rest, seph])
    return res


def valid_line_configurations(prev_conf, grid_size, side, max_shift):
    # given a line, returns the possible 'following' lines
    res = [()]
    if prev_conf:
        prev_start = prev_conf[0]
        prev_end = prev_conf[1]
        res += [(prev_start + x, prev_end - y)
                for x in range(maximum_shift + 1)
                for y in range(maximum_shift + 1)
                if prev_end - y >= prev_start + x]
        prev_length = prev_end - prev_start + 1
        if prev_length > max_shift:
            if side == right:
                for shift in range(1, max_shift + 1):
                    max_start = min(prev_start + shift, grid_size - 1)
                    max_end = min(prev_end + shift, grid_size - 1)
                    if max_end >= max_start:
                        res += [(max_start, max_end)]
                    if prev_end >= max_start:
                        res += [(max_start, prev_end)]
            elif side == left:
                for shift in range(1, max_shift + 1):
                    min_start = max(prev_start - shift, 0)
                    min_end = max(prev_end - shift, 0)
                    if min_end >= min_start:
                        res += [(min_start, min_end)]
                    if min_end >= prev_start:
                        res += [(prev_start, min_end)]
    return res


def shapesize(config):
    res = 0
    for line in config:
        if line:
            res += line[1] - line[0] + 1
    return res


def uniques(shapes, grid_size):
    shapesdef = tuple()
    while shapes:
        shape = shapes.pop(0)
        shapesdef += shape,
        r = list(range(-(grid_size // 2), 0)) + list(
            range(1, (grid_size // 2) + 1))
        identicalshapes = tuple()
        # left/right translations
        for i in r:
            ident = tuple()
            for j in shape:
                if j:
                    ident += (j[0] + i, j[1] + i),
                else:
                    ident += (),
            identicalshapes += (ident),
        # up/down translations
        emptylinesabove = 0
        while shape[emptylinesabove] == ():
            emptylinesabove += 1
        for ishape in ((shape,) + identicalshapes):
            for i in range(0, emptylinesabove):
                ident = list(ishape[i + 1:]) + [()] * (i + 1)
                identicalshapes += (tuple(ident)),
        i = grid_size - 1
        emptylinesbelow = 0
        while shape[i] == ():
            emptylinesbelow += 1
            i -= 1
        for ishape in ((shape,) + identicalshapes):
            for i in range(0, emptylinesbelow):
                ident = [()] * (i + 1) + list(ishape[:(grid_size - i - 1)])
                identicalshapes += (tuple(ident)),
        for s2 in identicalshapes:
            if s2 in shapes: shapes.remove(s2)
    return shapesdef


def centershape(shape, target):
    row, col = ndimage.measurements.center_of_mass(shape)
    row = int(round(row))
    col = int(round(col))
    shape = np.roll(np.roll(shape, target - row, 0), target - col)
    if shape[:, 0].any():
        shape = np.roll(shape, 1)
    if shape[:, -1].any():
        shape = np.roll(shape, -1)
    if shape[0, :].any():
        shape = np.roll(shape, 1, 0)
    if shape[-1, :].any():
        shape = np.roll(shape, -1, 0)
    if shape[:, 0].any() or shape[:, -1].any() or \
            shape[0, :].any() or shape[-1, :].any():
        print('Shape cannot be centered.')
    return shape


def notlinear(shapes):
    shapesdef = tuple()
    while shapes:
        shape = shapes.pop(0)
        vlinear = True
        filledlines = grid_size
        for line in shape:
            if line:
                if line[1] - line[0] > 0:
                    vlinear = False
            else:
                filledlines -= 1
        if not vlinear and filledlines > 1: shapesdef += shape,
    return shapesdef


def isshift(l1, l2):
    s1 = l1[0]
    s2 = l2[0]
    e1 = l1[1]
    e2 = l2[1]
    if s2 < s1 or e2 > e1: return True
    return False


def generateshapes(grid_size, minsize, maxsize, max_shift):
    center = grid_size // 2  # origin of shape
    centrallines = [[(x, y)] for x in range(grid_size)
                    for y in range(grid_size)
                    if x <= center <= y]  # only lines touching the origin
    configs = []
    j = 0
    print('Computing lines...')
    for side in (left, right):
        configstmp = centrallines[:]  # starting point
        for i in range(center):
            j += 1
            extended_configs = []
            print('%i/%i...' % (j, center * 2))
            # extend from the central line to the bottom
            for config in configstmp:
                if len(config) > 1 and config[-1] and isshift(config[-2],
                                                              config[-1]):
                    shift = max_shift
                else:
                    shift = 0
                next_line_configs = valid_line_configurations(config[-1],
                                                              grid_size,
                                                              side, shift)
                extended_configs.extend([config + [next_line_config]
                                         for next_line_config in
                                         next_line_configs])
            configstmp = extended_configs
            # extend from the bottom half-shape to the top
            extended_configs = []
            for config in configstmp:
                if config[0] and config[1] and isshift(config[1], config[0]):
                    shift = max_shift
                else:
                    shift = 0
                next_line_configs = valid_line_configurations(config[0],
                                                              grid_size,
                                                              not side, shift)
                extended_configs.extend([[next_line_config] + config
                                         for next_line_config in
                                         next_line_configs])
            configstmp = extended_configs
        configs += configstmp
    print('%i shapes' % len(configs))
    print('Removing shapes based on size...')
    shapes = [tuple(config) for config in configs
              if minsize <= shapesize(config) <= maxsize]
    print('%i shapes' % len(shapes))
    print('Removing linear shapes...')
    shapes = notlinear(shapes)
    print('%i shapes' % len(shapes))
    print('Removing duplicates...')
    shapes = list(set(shapes))
    print('%i shapes' % len(shapes))
    print('Removing duplicates by translation...')
    shapes = uniques(shapes, grid_size)
    print('%i shapes' % len(shapes))
    print('Converting lists of intervals to boolean arrays...')
    res = [intervalstoboolarray(config, grid_size) for config in shapes]
    print('Centering shapes...')
    res = [centershape(shape, center + 1) for shape in res]
    print('Sorting arrays...')
    res.sort(key=np.sum)
    return res


def com(shape):
    row, col = ndimage.measurements.center_of_mass(shape)
    row = grid_size // 2 - int(round(row))
    col = grid_size // 2 - int(round(col))
    return row, col


# shape parameters
grid_size = 3  # edge of square containing shapes
minimal_shape_surface = 1  # minimum size of object
maximal_shape_size = 9  # minimum size of object
maximum_shift = 1

# rendering parameters
bgcolor = 255  # background color
color = 0  # object color
sepcolor = 200  # separator color
septhickness = 1  # separator thickness
outputwidth = 150  # max width of image output

###############################################################################
right = True
left = False
conn8 = ndimage.generate_binary_structure(2, 2)

# generate shapes
shapes = generateshapes(grid_size, minimal_shape_surface, maximal_shape_size,
                        maximum_shift)

# add 1 pix and 2 pix shapes
p1 = np.zeros((grid_size + 2, grid_size + 2), bool)
p1[grid_size // 2 + 1, grid_size // 2 + 1] = True
p2 = np.zeros((grid_size + 2, grid_size + 2), bool)
p2[grid_size // 2 + 1:grid_size // 2 + 3, grid_size // 2 + 1] = True
p3 = np.zeros((grid_size + 2, grid_size + 2), bool)
p3[grid_size // 2 + 1, grid_size // 2 + 1:grid_size // 2 + 3] = True
shapes = [p1, p2, p3] + shapes

shapesarray = np.array([s[1:-1, 1:-1] for s in shapes])

# potrnps = ShapesLibrary(shapesarray, name="Patatoids 5x5")

np.savez("patatoids_3x3", shapesarray)
