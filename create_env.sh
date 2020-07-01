#!/usr/bin/env bash
#
# use the following command to run this script: . ./create_env.sh
# use the option `--gpu` followed by the CUDA version
#    e.g., `. ./create_env --gpu 10.1`
#
# Created on: Jul 01, 2020
# Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

GPU=false
if [[ "$1" == "--gpu" ]]; then
  GPU=true
  GPU_VERSION=$2
fi

if [[ "$OSTYPE" != "linux-gnu" && "$OSTYPE" != "darwin"* ]]; then
    echo "Operating system must be either linux or OS X"
    return 1
fi

command -v conda >/dev/null 2>&1 || { echo >&2 "I require anaconda but it's not installed.  Aborting."; return 1; }

conda update -n base conda --yes

packages=(
    coverage
    matplotlib
    nose
    numpy
    pandas
    pillow
    scipy
    sphinx
)

pytorch_packages=(
    pytorch
    torchvision
)

if [[ "$OSTYPE" != "linux-gnu" ]]; then
    if $GPU; then
        pytorch_packages+=("cudatoolkit=$GPU_VERSION")
    else
        pytorch_packages+=(cpuonly)
    fi
fi

conda_forge_packages=(
    humanize
    nibabel
    pytorch-lightning
)

conda create -n msseg python "${packages[@]}" -y
conda activate msseg
conda install -c pytorch "${pytorch_packages[@]}" -y
conda install -c conda-forge "${conda_forge_packages[@]}" -y
#conda install -c simpleitk simpleitk -y
pip install --upgrade torchio

python setup.py develop
