import functools


@functools.cache
def calc_inside_tiles_upper_bound(
    rows: int, cols: int, n_tiles: int, fixed_tile: tuple[int, int]
) -> int:
    """
    Solves the following problem:
        You can build a contiguous fence using n_tiles tiles on a grid of size rows x cols.
        One tile must be placed at the position fixed_tile, and the fence must intersect the first row and column.
        What is the maximum number of tiles that can be enclosed by the fence?
    This is useful for bounding the maximum enclosed area of a polyomino fence that places a polyomino
    covering fixed_tile. It can eliminate placements that yield a smaller enclosed area than desired.
    """
    assert rows >= 3 and cols >= 3
    fr, fc = fixed_tile
    result = max(
        (
            calc_inside_tiles_upper_bound_fixed_dims(h, w, n_tiles, fixed_tile)
            for h in range(3, rows + 1)
            for w in range(3, cols + 1)
        ),
        default=-1,
    )
    return result


@functools.cache
def calc_inside_tiles_upper_bound_fixed_dims(
    h: int, w: int, n_tiles: int, fixed_tile: tuple[int, int]
) -> int:
    """
    Returns an upper bound for the area of the best fence of main dimensions (h,w)
    that uses <= n_tiles tiles and places a tile at fixed_tile.
    Any such fence must be rectangular or L-shaped, with a possible inner or outer protrusion connecting to fixed_tile.
    """
    assert h >= 3 and w >= 3
    fr, fc = fixed_tile

    initial_perimeter = 2 * (h + w) - 4
    if initial_perimeter > n_tiles:
        return -1

    if fr >= h or fc >= w:  # Outer protrusion
        initial_area = (h - 2) * (w - 2)
        return initial_area
    else:  # Inner protrusion
        # Build an initial L-shaped fence taking up (0,0) to (h,w) with inner corner at fixed_tile
        dr = min(h - fr - 1, fr)
        dc = min(w - fc - 1, fc)
        initial_area = (h - 2) * (w - 2) - (dr * dc)

        # Shift the longer inner wall as close to the edge as possible. Each shift costs 1 tile.
        longer_inner_wall, shorter_inner_wall = max(dr, dc), min(dr, dc)
        slack_tiles = n_tiles - initial_perimeter
        wall_shift = min(slack_tiles, shorter_inner_wall)
        area = initial_area + wall_shift * (longer_inner_wall - 1)

        return area
