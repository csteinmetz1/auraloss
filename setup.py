#!/usr/bin/env python3
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path
from setuptools import setup, find_packages

NAME = 'auraloss'
DESCRIPTION = 'Audio-focused loss functions in PyTorch'
URL = 'https://github.com/csteinmetz1/auraloss'
EMAIL = 'c.j.steinmetz@qmul.ac.uk'
AUTHOR = 'Christian Steinmetz'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = "0.1.6"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['auraloss'],
    install_requires=['torch', 'scipy', 'numpy', 'librosa', 'matplotlib'],
    extras_require={"test": ['resampy']},
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering',
    ],
)