from typing import List, Tuple

Tile = Tuple[int, int]


class Polyomino:
    """
    Base Polyomino Class

    For pentominoes, uses the first naming convention found in
    the pentomino wiki page: https://en.wikipedia.org/wiki/Pentomino
    """

    name = None
    default_tiles: List[Tile] = []

    def __init__(self):
        self.tiles = self.default_tiles

    def rotate(self, k: int) -> "Polyomino":
        k = k % 4
        for _ in range(k):
            self.tiles = [(y, -x) for (x, y) in self.tiles]
        return self

    def reflect_horizontally(self) -> "Polyomino":
        self.tiles = [(-x, y) for (x, y) in self.tiles]
        return self

    def reflect_vertically(self) -> "Polyomino":
        self.rotate(2)
        self.reflect_horizontally()
        return self

    def recenter(self, new_center: Tile) -> "Polyomino":
        cx, cy = new_center
        self.tiles = [(x - cx, y - cy) for (x, y) in self.tiles]
        return self

    def print_in_grid(self, size: int = 7):
        grid = [[" " for _ in range(size)] for _ in range(size)]

        offset = size // 2

        for x, y in self.tiles:
            gx, gy = x + offset, y + offset
            if 0 <= gx < size and 0 <= gy < size:
                grid[gy][gx] = "x" if (x, y) == (0, 0) else "o"

        for row in reversed(grid):
            print(" ".join(row))


class F5(Polyomino):
    """
      o o
    x o
      o
    """

    name = "F"
    default_tiles = [(0, 0), (1, 0), (1, -1), (1, 1), (2, 1)]


class I5(Polyomino):
    """
    o
    o
    o
    o
    x
    """

    name = "I"
    default_tiles = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


class L5(Polyomino):
    """
    o
    o
    o
    x o
    """

    name = "L"
    default_tiles = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]


class N5(Polyomino):
    """
      o
      o
    o o
    x
    """

    name = "N"
    default_tiles = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]


class P5(Polyomino):
    """
    o o
    o o
    x
    """

    name = "P"
    default_tiles = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]


class T5(Polyomino):
    """
    o o o
      o
      x
    """

    name = "T"
    default_tiles = [(0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)]


class U5(Polyomino):
    """
    o   o
    x o o
    """

    name = "U"
    default_tiles = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]


class V5(Polyomino):
    """
        o
        o
    x o o
    """

    name = "V"
    default_tiles = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


class W5(Polyomino):
    """
        o
      o o
    x o
    """

    name = "W"
    default_tiles = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]


class X5(Polyomino):
    """
      o
    o o o
      x
    """

    name = "X"
    default_tiles = [(0, 0), (0, 1), (-1, 1), (1, 1), (0, 2)]


class Y5(Polyomino):
    """
      o
    o o
      o
      x
    """

    name = "Y"
    default_tiles = [(0, 0), (0, 1), (0, 2), (-1, 2), (0, 3)]


class Z5(Polyomino):
    """
    o o
      o
      x o
    """

    name = "Z"
    default_tiles = [(0, 0), (1, 0), (0, 1), (0, 2), (-1, 2)]
