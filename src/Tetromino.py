from Polyomino import Polyomino

class I4(Polyomino):
    """
    o
    o
    o
    x
    """

    name = "I"
    default_tiles = [(0, 0), (0, 1), (0, 2), (0, 3)]


class O4(Polyomino):
    """
    o o
    x o
    """

    name = "O"
    default_tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]


class T4(Polyomino):
    """
    o o o
      x
    """

    name = "T"
    default_tiles = [(0, 0), (-1, 1), (0, 1), (1, 1)]


class L4(Polyomino):
    """
    o
    o
    x o
    """

    name = "L"
    default_tiles = [(0, 0), (0, 1), (0, 2), (1, 0)]


class N4(Polyomino):
    """
      o o
    x o
    """

    name = "N"
    default_tiles = [(0, 0), (1, 0), (1, 1), (2, 1)]

ALL_TETROMINOES = [
    I4(),
    O4(),
    T4(),
    L4(),
    N4(),
]