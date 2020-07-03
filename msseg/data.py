#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
msseg.data

general file/data-handling operations

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: Jul 01, 2020
"""

__all__ = ['csv_to_subjectlist',
           'glob_ext']

from typing import *

import contextlib
from glob import glob
import os
from os.path import join

import pandas as pd

with open(os.devnull, "w") as f:
    with contextlib.redirect_stdout(f):
        import torchio


def glob_ext(path:str, ext:str='*.nii*') -> List[str]:
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(join(path, ext)))
    return fns


def csv_to_subjectlist(filename:str) -> List[torchio.Subject]:
    """ Convert a csv file to a list of torchio subjects

    Args:
        filename: Path to csv file formatted with
            `subject` in a column, describing the
            id/name of the subject (must be unique).
            Row will fill in the filenames per type.
            Other columns headers must be one of:
            ct, flair, label, pd, t1, t1c, t2
            (`label` should correspond to a
             segmentation mask)

    Returns:
        subject_list (List[torchio.Subject]):
    """
    valid_names = ['ct', 'flair', 'label', 'pd', 't1', 't1c', 't2']
    df = pd.read_csv(filename, index_col='subject')
    names = df.columns.to_list()
    if any([name not in valid_names for name in names]):
        raise ValueError(f'Column name needs to be in {valid_names}')

    subject_list = []
    for row in df.iterrows():
        subject_name = row[0]
        images = {}
        for name in names:
            type = torchio.LABEL if name == 'label' else torchio.INTENSITY
            fn = row[1][name]
            images[name] = torchio.Image(fn, type=type)
        subject = torchio.Subject(
            name=subject_name,
            **images)
        subject_list.append(subject)

    return subject_list
