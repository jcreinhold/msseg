tiramisu
========

A PyTorch implementation of a 2D and 3D Tiramisu convolutional neural network.

This package is primarily used for multiple sclerosis (MS) lesion segmentation; specifically, T2 lesions in the brain.

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io)
and the other students and researchers of the
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

Requirements
------------

- nibabel >= 3.1
- numpy >= 1.18
- pytorch >= 1.5
- pytorch_lightning >= 0.8.4
- torchio >= 0.17

Installation
------------

From inside this directory, run:

    python setup.py install

or (if you'd like to make updates to the package)

    python setup.py develop

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests

References
---------------

[1] Jégou, Simon, et al. "The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation."
CVPR. 2017.

[2] Zhang, Huahong, et al. "Multiple sclerosis lesion segmentation with Tiramisu and 2.5D stacked slices." International
Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.
