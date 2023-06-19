from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup(
    author='Patrick Aschenbrenner',
    author_email='patrick.aschenbrenner@student.uibk.ac.at',
    name='cais_rtod',
    version='0.0.2',
    description='Real-time object detection for images in CAIS simulations.',
    url='https://github.com/PatrickAschenbrenner/cais_rtod',
    packages=find_packages(where='src', exclude=('test*')),
    package_dir={'': 'src'},
)
