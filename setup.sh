#!/bin/bash

# pre-steps:
# git clone https://github.com/nathanlct/FCGF.git
# cd FCGF
# ./setup.sh

pip install -r requirements.txt --user
pip install open3d --user
pip install future-fstrings --user
pip install -U MinkowskiEngine --user
pip install tensorflow --user

rm Benchmark_MVA.zip
rm npm_data.tar.xz

wget https://github.com/nathanlct/FCGF/raw/master/Benchmark_MVA.zip
unzip Benchmark_MVA.zip
mv Benchmark_MVA dataset_small

wget https://github.com/nathanlct/FCGF/raw/master/npm_data.tar.xz
tar xf npm_data.tar.xz 
mv npm_data dataset

python generate_features.py