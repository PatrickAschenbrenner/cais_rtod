from setuptools import setup
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
    version='0.0.1',
    description='Real-time object detection for images in CAIS simulations.',
    packages=['cais_rtod', 'cais_rtod.hog', 'cais_rtod.svm', 'cais_rtod.yolo',
              'cais_rtod.detector'],
    package_dir={'': 'src'}
)
