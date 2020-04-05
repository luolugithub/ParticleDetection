#!/usr/bin/env python

from argparse import ArgumentParser
from sys import exit
from re import sub

from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from numpy import percentile, ones_like, array_str

from spade.spade.detection_2d import spade2d


args = ArgumentParser(description='Command-line utility for SPADE.\n'
                                  'It is not recommended to use this '
                                  'interface for batch image processing as '
                                  'each process initialization is slow. Use '
                                  'a python script instead.')
args.add_argument('INPUT_FILE', type=str, help='The image to process. If 3D,'
                                               'the maximum intensity '
                                               'projection will be used.')
args.add_argument('THRESHOLD', type=float, help='The SPADE threshold to use.')
args.add_argument('--compute-mask', '-c',
                  dest='compute_mask',
                  action='store_true',
                  default=False,
                  help='Use Otsu\'s thresholding method to compute the image '
                       'mask.\n Ignored if a mask file is specified.')
args.add_argument('--normalize-mean', '-n',
                  dest='normalize_mean',
                  action='store_true',
                  help='Normalize image mean before detection.',
                  default=False)
args.add_argument('--output', '-o', type=str, default='/tmp/spade_result.png',
                  help='The output image. Each particle will have a '
                       'different grayscale value.\n'
                       '/tmp/spade_result.png by default.')
args.add_argument('--data-binding-function', '-f',
                  dest='data_binding_function',
                  type=str,
                  default=None,
                  help='The data binding function to use.\n'
                       'mean_difference by default. Possible choices: '
                       'max_min_difference, median_difference, '
                       'bhattacharryya_distance, '
                       'pseudo_bhattacharryya_distance and t_test')
args.add_argument('--mask', '-m',
                  type=str,
                  default=None,
                  help='An image used as a mask for the detection.\n'
                       'Any pixels above zero will be interpreted as '
                       'positive.')
args.add_argument('--shapes-library', '-s',
                  dest='shapes_library',
                  type=str,
                  default='potatoids5x5_smallest3px',
                  help='The shapes library to use.'
                       'Default is potatoids5x5_smallest3px. Possibles '
                       'choices: potatoids[GRID]_smallest[PX]px[SUFFIX]) '
                       'where GRID=5x5 or 3x3; '
                       '      PX=1 to 4; '
                       '      SUFFIX=nothing, _distantrings or _thickrings')
args.add_argument('--restrict', '-r',
                  type=float,
                  default=0,
                  help='Restrict the shapes potential centers to the n '
                       'percent brightest pixels.')
args = args.parse_args()


if args.data_binding_function is not None:
    try:
        exec('from spade.data_binding import {} '
             'as data_binding_function'.format(args.data_binding_function))
    except ImportError:
        print('Cannot find data binding function {}.'.format(
            args.data_binding_function))
        exit(1)
else:
    data_binding_function = None

try:
    exec('from spade.shapes.examples import {} as shapes_library'.format(
        args.shapes_library))
except ImportError:
    print('Cannot find shapes library {}.'.format(args.shapes_library))
    exit(1)
print('Using shapes library {}.'.format(args.shapes_library))

try:
    image = imread(args.INPUT_FILE)
except FileNotFoundError:
    print('Cannot find image file {}.'.format(args.INPUT_FILE))
    exit(1)

if image.ndim > 2:
    print('3D image passed as input, using maximum intensity projection.')
    image = image.max(axis=0)

if args.mask is None:
    if args.compute_mask:
        print('Computing mask using Otsu\'s thresholding.')
        mask = image > threshold_otsu(image)
    else:
        print('No mask used.')
        mask = ones_like(image, dtype=bool)
else:
    try:
        mask = imread(args.mask).astype(bool)
    except FileNotFoundError:
        print('Cannot find mask file {}.'.format(args.mask))
        exit(1)
    print('Using mask file {}.'.format(args.mask))

if args.restrict:
    print('Restricting detection to the {}% brightest pixels.'.format(
        args.restrict))
    potential_centers = image > percentile(image[mask], 100 - args.restrict)
else:
    potential_centers = None

if args.normalize_mean:
    print('Normalizing image mean.')
    image = image / image.mean()

print('SPADEing the image...')
result, particles = spade2d(image=image,
                            shapes_library=shapes_library,
                            threshold=args.THRESHOLD,
                            mask=mask,
                            data_binding_function=None,
                            potential_centers=potential_centers,
                            return_candidates_list=True)
print('{} particles detected.'.format(result.max()))
print('    Y    X  Shape_ID')
print(' ' + sub('[\[\]]', '', array_str(particles)))

imsave(args.output, result)
print('Results written to {}'.format(args.output))
