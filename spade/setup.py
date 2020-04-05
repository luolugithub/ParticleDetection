from distutils.core import setup
from distutils.command.install import INSTALL_SCHEMES
from os import path

for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='small-particle-detection',
    version='0.0.3',
    packages=['spade', 'spade.shapes'],
    url='https://gitlab.inria.fr/ncedilni/spade',
    license='CeCILL',
    author='Nicolas Cedilnik',
    author_email='nicolas.cedilnik@inria.fr',
    description='Small Particle Detection, a shapes library-based object '
                'detector.',
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, '
        'version 2.1 (CeCILL-2.1)',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='image, image processing, scientific research',
    install_requires=['numpy'],
    package_data={'spade.shapes': ['*.npz']},
    include_package_data=True
)
