#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.io

general file operations

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['glob_ext']

from typing import *

from glob import glob
import os


def glob_ext(path:str, ext:str='*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, ext)))
    return fns
