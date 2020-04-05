SPADE: Small PArticle DEtection
===============================

.. image:: http://www-sop.inria.fr/morpheme/images/logo2.png
   :width: 200 px
   :target: http://www-sop.inria.fr/morpheme/team.html

An algorithm primarily design to detect objects whose sizes aren't larger a
few pixels (particles) on fluorescence microscopy images.

It is an simplified version of marked point process.


Requirements
============
SPADE has only been tested with Python 3.5 but should work with older versions.
The only strict requirement is the package `numpy`.
In order to use SPADE to its full extent, we recommend installing the
`scikit-image` package and the full `scipy` ecosystem.


Usage
=====

In a python script
------------------

You can install the package using PIP. The main function, `spade2d`, is in the
`spade.detection_2d` module. It takes an image, a shapes library and a
threshold as input. A typical usage would look like like

.. code:: python

    from skimage.io import imread, imshow
    from skimage.filters import threshold_otsu
    from skimage.color import label2rgb
    from numpy import percentile
    import matplotlib.pyplot as plt

    from spade.detection_2d import spade2d
    from spade.shapes.examples import potatoids5x5_smallest4px


    # Load the example image.
    image = imread("example_2d.png")

    # Separate cell image from background, using by Otsu's thresholding method.
    cell = image > threshold_otsu(image)

    # Focus on brightest pixels only
    potential_centers = image > percentile(image[cell], 99)

    # Detect particles.
    particles = spade2d(image=image,
                        shapes_library=potatoids5x5_smallest4px,
                        threshold=20,
                        potential_centers=potential_centers,
                        mask=cell)

    # Show detected particles as overlay on our original image.
    imshow(label2rgb(particles, image, bg_label=0))
    plt.show()


See the examples in the `example` folder of the source dist for more
information. Note that `matplotlib` and `skimage` are not required for SPADE
to work.

From the command line
---------------------

Also provided is a command line interface, that can be used as following:

.. code:: bash

    python -m spade.cli INPUT_IMAGE THRESHOLD -o /path/to/output_image.png

Launch this command with `-h` for more information on its usage.



Authors & Acknowledgments
=========================


SPADE was developed by `Nicolas Cedilnik
<mailto:nicoco@nicoco.fr>`_, `Éric Debreuve
<http://www.i3s.unice.fr/~debreuve/>`_ and `Xavier Descombes
<http://www-sop.inria.fr/members/Xavier.Descombes/>`_ at the `MORPHEME team
<http://www-sop.inria.fr/morpheme/team.html>`_.

If you use SPADE for your scientific work, please include this `bibtex` (or
equivalent) entry:

.. code:: latex

    @misc{cedilnik2016spade,
      title={SPADE: Small particle detection}
      author={Cedilnik, Nicolas and Debreuve, Éric and Descombes, Xavier}
      url={https://pypi.python.org/pypi/small-particle-detection}
      year={2016}
    }

License
=======
SPADE is released under the `CeCILL-2.1 licence
<http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.txt>`_.
