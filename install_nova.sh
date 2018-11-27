#!/usr/bin/env bash

env_name="nova"

# create a conda environment
# install packages from environment.yml
conda env create --name $env_name --file environment.yml

# install packages that are not listed in conda
# change to your anaconda installation path
conda_path=~/anaconda3

subdir=envs/$env_name/bin
pip_path=$conda_path/$subdir
sudo $pip_path/pip install gmpy2==2.1.0a1 --ignore-installed --no-binary ":all:"
sudo $pip_path/pip install pplpy
