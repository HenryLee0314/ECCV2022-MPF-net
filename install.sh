#!/bin/bash
cd ./models/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
CC=g++-10 CXX=g++-10 python3 setup.py install --user

cd ..
