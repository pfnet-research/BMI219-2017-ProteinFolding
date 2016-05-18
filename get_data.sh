#!/usr/bin/env bash

if [ ! -e data ]
then
    mkdir data
fi

wget http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz -P data
wget http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz -P data

cd data
gunzip cb513+profile_split1.npy.gz
gunzip cullpdb+profile_6133_filtered.npy.gz
