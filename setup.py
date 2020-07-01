#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs msseg package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: July 01, 2020
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='msseg',
    version='0.1.0',
    description="MS brain T2 lesion segmentation",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/msseg',
    license=license,
    packages=find_packages(exclude=('tests', 'tutorials', 'docs')),
    keywords="medical image segmentation",
)

setup(install_requires=['numpy',
                        'torch',
                        'torchvision'], **args)
