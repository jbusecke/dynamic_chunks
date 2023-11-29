# dynamic_chunks

Determine appropriate chunk sizes for a given xarray dataset based on target chunk size and 'chunk aspect ratio'

The chunk aspect ratio describes the amount of chunks along a given dimension. Take a dataset with two dimensions (`a` and `b`). A chunk aspect ratio `{'a':2, 'b':1}` means that the number of total chunks along `b` is twice that of `b`.
This concept was inspired by a discussion with [Rich Signell](https://github.com/rsignell-usgs) at Scipy '23. The idea is that one might want to optimize the chunking of a dataset to make e.g. an operation along time n times slower than an operation along spatial dimensions.


## Usage
`dynamic_chunks` implements several algorithms with the input signature

```python
chunk_dict = choosen_algorithm(ds, desired_chunksize, target_aspect_ratio, size_tolerance)
```
Lets demonstrate this with the xarray example dataset

```python
import xarray as xr
from dynamic_chunks.algorithms import even_divisor_algo
ds = xr.tutorial.open_dataset("rasm").load()
```

### Rechunking with even divisors along dimensions
`dynamic_chunks.algorithms.even_divisor_algo` restricts possible rechunking options to only even divisors of each chunked dimension.

The simplest case is rechunking a dataset along a single dimension (the `target_chunk_ratio` only contains a single dimension and the value does not matter)

```python
# Aim for 1MB chunks along time only with a tolerance of 0.2 (so we will accept chunks from 0.8-1.2 MB)
even_divisor_algo(ds, '1MB', {'time':1}, 0.5)
```
gives 
```
{'time': 2, 'y': 205, 'x': 275}
```

Lets test that really quick
```python
ds.chunk({'time': 2, 'y': 205, 'x': 275})
```
<img width="646" alt="image" src="https://github.com/jbusecke/dynamic_chunks/assets/14314623/75c88672-965c-4748-af5b-84517e04776c">

Nice this gave us ~800KB chunks.

### Chunk along multiple dimensions
Lets chunk along all dimensions of the dataset and aim to have the same number of chunks along each dimension:
```python
chunks = even_divisor_algo(ds, '100KiB', {'x':1, 'y':1, 'time':1}, 0.3)
ds.chunk(chunks)
```
<img width="648" alt="image" src="https://github.com/jbusecke/dynamic_chunks/assets/14314623/eb866582-f41b-4984-b88f-1091b007ee26">

Ok nice we have chunks of the desired chunksize (~100KB). 

> Note that I chose a very small chunksize here for demonstration purposes, in practice you should adjust the chunksize to your specific use case (more reading [here](https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes)).

Finally lets check the total number of chunks along each dimension
```python
{k:len(v) for k,v in ds.chunk(chunks).chunks.items()}
```
```
{'time': 6, 'y': 5, 'x': 5}
```
Not bad! Please note that due to the requirements of even divisors the resulting ratio of chunks can significantly differ from the target you provided. You should experiment with different values for size, tolerance to arrive at a chunking scheme that works for you.

### Keeping one dimension unchunked

In many cases you simply do not want to chunk along a given dimension at all. You can use the sentinel value `-1` in `target_chunk_ratio` to keep specific dimensions unchunked:

```python
chunks = even_divisor_algo(ds, '100KiB', {'x':1, 'y':1, 'time':-1}, 0.3)
chunks
```
gives
```
{'time': 36, 'y': 41, 'x': 11}
```
which means the `time` dimension is only a single chunk




## Developer Guide

Set up your development environment with `conda`:

```
conda create --name dynamic_rechunking python=3.10 pip
conda activate dynamic_rechunking
pip install -e ".[test]"
```
