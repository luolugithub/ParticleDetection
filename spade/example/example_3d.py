from skimage.io import imread
from skimage.filters import threshold_otsu
from numpy import percentile

from spade.detection_3d import spade3d
from spade.shapes.examples import potatoids5x5_smallest4px

from imsave3d import imsave3d_overlay

# Load the example image.
image = imread("example_3d.tif")

# Separate cell image from background, using by Otsu's thresholding method.
cell = image > threshold_otsu(image)

# Focus on brightest pixels only
potential_centers = image > percentile(image[cell], 90)

# Detect particles.
particles = spade3d(image3d=image,
                    shapes_library=potatoids5x5_smallest4px,
                    threshold=118,
                    potential_centers=potential_centers,
                    mask=cell)

# Save result image. Open it with FIJI or your favorite TIFF stack viewer!
imsave3d_overlay('/tmp/spade3d_result.tif', image, cell, particles)
