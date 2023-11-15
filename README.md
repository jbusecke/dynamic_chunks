# dynamic_chunks

Determine appropriate chunk sizes for a given xarray dataset based on target chunk size and 'chunk aspect ratio'

The chunk aspect ratio describes the amount of chunks along a given dimension. Take a dataset with two dimensions (`a` and `b`). A chunk aspect ratio `{'a':2, 'b':1}` means that the number of total chunks along `b` is twice that of `b`.
This concept was inspired by a discussion with [Rich Signell](https://github.com/rsignell-usgs) at Scipy '23. The idea is that one might want to optimize the chunking of a dataset to make e.g. an operation along time n times slower than an operation along spatial dimensions.

## Useage
TBW

## Developer Guide

Set up your development environment with `conda`:

```
conda create --name dynamic_rechunking python=3.10 pip
conda activate dynamic_rechunking
pip install -e ".[test]"
```
