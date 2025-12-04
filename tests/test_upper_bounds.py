from src.upper_bounds import (
    calc_inside_tiles_upper_bound,
    calc_inside_tiles_upper_bound_fixed_dims,
)
import pytest


@pytest.mark.parametrize(
    "rows, cols, n_tiles, fixed_tile, expected",
    [
        (3, 3, 8, (0, 0), 1),
        (3, 3, 8, (1, 1), 0),
        (4, 4, 12, (0, 0), 4),
        (5, 5, 18, (2, 2), 7),
        (8, 7, 26, (4, 4), 24),
        (8, 7, 27, (4, 4), 26),
        (10, 7, 27, (4, 4), 27),
        (8, 8, 25, (7, 2), 25),
    ],
)
def test_calc_inside_tiles_upper_bound(rows, cols, n_tiles, fixed_tile, expected):
    result = calc_inside_tiles_upper_bound(rows, cols, n_tiles, fixed_tile)
    assert result == expected


@pytest.mark.parametrize(
    "h, w, n_tiles, fixed_tile, expected",
    [
        (3, 3, 8, (0, 0), 1),
        (3, 3, 8, (1, 1), 0),
        (4, 4, 12, (0, 0), 4),
        (8, 7, 26, (4, 4), 24),
        (8, 7, 27, (4, 4), 26),
    ],
)
def test_calc_inside_tiles_upper_bound_fixed_dims(h, w, n_tiles, fixed_tile, expected):
    result = calc_inside_tiles_upper_bound_fixed_dims(h, w, n_tiles, fixed_tile)
    assert result == expected
