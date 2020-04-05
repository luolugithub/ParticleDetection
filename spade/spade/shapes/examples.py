import numpy as np
from spade.spade.shapes.library import ShapesLibrary
from os.path import dirname, join

local_dir = dirname(__file__)
starts = (0, 1, 3, 7)
sizes = ('5x5', '3x3')
potatoids_5x5_array = np.load(join(local_dir,
                                   'potatoids_5x5.npz'))['arr_0.npy']

for i, p in enumerate(starts, start=1):
    for size in sizes:
        array = np.load(join(local_dir,
                             'potatoids_{}.npz'.format(size)))['arr_0.npy']
        exec("potatoids{0}_smallest{2}px = ShapesLibrary(array[{1}:],"
             "name='Potatoids {0}; smallest={2}px', sort=False)".format(size,
                                                                        p, i))
        exec("potatoids{0}_smallest{2}px_distantrings = ShapesLibrary("
             "array[{1}:], ring_distance=1,"
             "name='Potatoids {0}; smallest={2}px; distant rings', "
             "sort=False)".format(size, p, i))
        exec("potatoids{0}_smallest{2}px_thickrings = ShapesLibrary("
             "array[{1}:], ring_thickness=2,"
             "name='Potatoids {0}; smallest={2}px; thick rings', "
             "sort=False)".format(size, p, i))

del array

potatoids_5x5 = potatoids5x5_smallest3px
potatoids_3x3 = potatoids3x3_smallest3px
