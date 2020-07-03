#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.util

miscellaneous functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['n_dirname']

import os


def n_dirname(path:str, n:int) -> str:
    """ return n-th dirname from basename """
    dirname = path
    for _ in range(n):
        dirname = os.path.dirname(dirname)
    return dirname
