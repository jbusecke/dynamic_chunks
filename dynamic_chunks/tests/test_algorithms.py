from typing import Dict

import dask.array as dsa
import pytest
import xarray as xr


from dynamic_chunks.algorithms import (
    even_divisor_algo,
    iterative_ratio_increase_algo,
    NoMatchingChunks
)


def _create_ds(dims_shape: Dict[str, int]) -> xr.Dataset:
    return xr.DataArray(
        dsa.random.random(list(dims_shape.values())),
        dims=list(dims_shape.keys()),
    ).to_dataset(name="data")


@pytest.mark.parametrize(
    ("dims_shape", "target_chunks_aspect_ratio", "expected_target_chunks"),
    [
        # make sure that for the same dataset we get smaller chunksize along
        # a dimension if the ratio is larger
        (
            {"x": 300, "y": 300, "z": 300},
            {"x": 1, "y": 1, "z": 10},
            {"x": 100, "y": 100, "z": 12},
        ),
        (
            {"x": 300, "y": 300, "z": 300},
            {"x": 10, "y": 1, "z": 1},
            {"x": 12, "y": 100, "z": 100},
        ),
        # test the special case where we want to just chunk along a single dimension
        (
            {"x": 100, "y": 300, "z": 400},
            {"x": -1, "y": -1, "z": 1},
            {"x": 100, "y": 300, "z": 4},
        ),
    ],
)
def test_dynamic_rechunking(dims_shape, target_chunks_aspect_ratio, expected_target_chunks):
    ds = _create_ds(dims_shape)
    target_chunks = even_divisor_algo(
        ds, 1e6, target_chunks_aspect_ratio=target_chunks_aspect_ratio, size_tolerance=0.2
    )
    print(target_chunks)
    print(expected_target_chunks)
    for dim, chunks in expected_target_chunks.items():
        assert target_chunks[dim] == chunks

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_nbytes_str_input(algo):
    ds = _create_ds({"x": 100, "y": 100, "z": 100})
    target_chunks_aspect_ratio = {"x": 1, "y": 1, "z": 1}
    target_chunks_int = algo(
        ds, 1e6, target_chunks_aspect_ratio=target_chunks_aspect_ratio, size_tolerance=0.2
    )
    target_chunks_str = algo(
        ds, "1MB", target_chunks_aspect_ratio=target_chunks_aspect_ratio, size_tolerance=0.2
    )
    for dim in target_chunks_aspect_ratio.keys():
        assert target_chunks_int[dim] == target_chunks_str[dim]


def test_maintain_ratio():
    """Confirm that for a given ratio with two differently sized datasets we
    maintain a constant ratio between total number of chunks"""
    ds_equal = _create_ds({"x": 64, "y": 64})
    ds_long = _create_ds({"x": 64, "y": 256})

    for ds in [ds_equal, ds_long]:
        print(ds)
        target_chunks = even_divisor_algo(
            ds, 1e4, target_chunks_aspect_ratio={"x": 1, "y": 4}, size_tolerance=0.2
        )
        ds_rechunked = ds.chunk(target_chunks)
        assert len(ds_rechunked.chunks["y"]) / len(ds_rechunked.chunks["x"]) == 4


@pytest.mark.parametrize(
    "target_chunks_aspect_ratio", [{"x": 1, "y": -1, "z": 10}, {"x": 6, "y": -1, "z": 2}]
)  # always keep y unchunked, and vary the others
@pytest.mark.parametrize("target_chunk_nbytes", [1e6, 5e6])
@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_skip_dimension(target_chunks_aspect_ratio, target_chunk_nbytes, algo):
    ds = _create_ds({"x": 100, "y": 200, "z": 300})
    # Mark dimension as 'not-to-chunk' with -1
    target_chunks = algo(
        ds,
        target_chunk_nbytes,
        target_chunks_aspect_ratio=target_chunks_aspect_ratio,
        size_tolerance=0.2,
    )
    assert target_chunks["y"] == len(ds["y"])


@pytest.mark.parametrize("default_ratio", [-1, 1])
@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_missing_dimensions(default_ratio, algo):
    ds = _create_ds({"x": 100, "y": 200, "z": 300})
    # Test that a warning is raised
    target_chunk_nbytes = 5e6
    msg = "are not specified in target_chunks_aspect_ratio.Setting default value of"
    with pytest.warns(UserWarning, match=msg):
        chunks_from_default = algo(
            ds,
            target_chunk_nbytes,
            target_chunks_aspect_ratio={"x": 1, "z": 10},
            size_tolerance=0.2,
            default_ratio=default_ratio,
        )
    chunks_explicit = algo(
        ds,
        target_chunk_nbytes,
        target_chunks_aspect_ratio={"x": 1, "y": default_ratio, "z": 10},
        size_tolerance=0.2,
    )
    assert chunks_from_default == chunks_explicit

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_permuted_dimensions(algo):
    ds = _create_ds({"x": 100, "y": 200, "z": 300})
    size_tolerance = 0.6
    target_chunk_size = 5e5
    target_chunks = algo(
        ds,
        target_chunk_size,
        target_chunks_aspect_ratio={"x": 1, "y": 2, "z": 10},
        size_tolerance=size_tolerance,
    )
    target_chunks_permuted = algo(
        ds,
        target_chunk_size,
        target_chunks_aspect_ratio={
            "z": 10,
            "y": 2,
            "x": 1,
        },
        size_tolerance=size_tolerance,
    )
    assert target_chunks == target_chunks_permuted

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_error_extra_dimensions_not_allowed(algo):
    ds = _create_ds({"x": 100, "y": 200, "z": 300})
    msg = "target_chunks_aspect_ratio contains dimensions not present in dataset."
    with pytest.raises(ValueError, match=msg):
        algo(
            ds,
            1e6,
            target_chunks_aspect_ratio={"x": 1, "y_other_name": 1, "y": 1, "z": 10},
            size_tolerance=0.2,
        )

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_extra_dimensions_allowed(algo):
    ds = _create_ds({"x": 100, "y": 200, "z": 300})
    with pytest.warns(UserWarning, match="Trimming dimensions"):
        chunks_with_extra = algo(
            ds,
            5e5,
            target_chunks_aspect_ratio={"x": 1, "y_other_name": 1, "y": 1, "z": 10},
            size_tolerance=0.2,
            allow_extra_dims=True,
        )
    chunks_without_extra = algo(
        ds,
        5e5,
        target_chunks_aspect_ratio={"x": 1, "y": 1, "z": 10},
        size_tolerance=0.2,
    )
    assert chunks_with_extra == chunks_without_extra

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_non_int_ratio_input(algo):
    ds = _create_ds({"x": 1, "y": 2, "z": 3})
    with pytest.raises(ValueError, match="Ratio value must be an integer. Got 1.5 for dimension y"):
        algo(
            ds,
            1e6,
            target_chunks_aspect_ratio={"x": 1, "y": 1.5, "z": 10},
            size_tolerance=0.2,
        )

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_large_negative_ratio_input(algo):
    ds = _create_ds({"x": 1, "y": 2, "z": 3})
    with pytest.raises(
        ValueError, match="Ratio value can only be larger than 0 or -1. Got -100 for dimension y"
    ):
        algo(
            ds,
            1e6,
            target_chunks_aspect_ratio={"x": 1, "y": -100, "z": 10},
            size_tolerance=0.2,
        )

def test_algo_comparison():
    """test that we get the same result from both algorithms for a known simple case"""
    ds = _create_ds({"x": 100, "y": 100, "z": 100})
    target_chunk_size = 4e5
    target_chunks_aspect_ratio = {"x": -1, "y": 2, "z": 10}
    size_tolerance = 0.01
    chunks_a = even_divisor_algo(
        ds,
        target_chunk_size,
        target_chunks_aspect_ratio=target_chunks_aspect_ratio,
        size_tolerance=size_tolerance,
    )
    chunks_b = iterative_ratio_increase_algo(
        ds,
        target_chunk_size,
        target_chunks_aspect_ratio=target_chunks_aspect_ratio,
        size_tolerance=size_tolerance,
    )
    assert chunks_a == chunks_b

@pytest.mark.parametrize("algo", [iterative_ratio_increase_algo, even_divisor_algo])
def test_algo_exception(algo):
    """Test that each of the algos raises our custom exception when we give some totally unsolvable parameters"""
    with pytest.raises(NoMatchingChunks):
        ds = _create_ds({"x": 10, "y": 10, "z": 10})
        target_chunk_size = 4e10
        target_chunks_aspect_ratio = {"x": -1, "y": 2, "z": 10}
        size_tolerance = 0.01
        chunks_a = algo(
            ds,
            target_chunk_size,
            target_chunks_aspect_ratio=target_chunks_aspect_ratio,
            size_tolerance=size_tolerance,
        )
        
