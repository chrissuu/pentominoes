from polyomino import Polyomino


class F5(Polyomino):
    """
      o o
    x o
      o
    """

    name = "F"
    default_tiles = [(0, 0), (1, 0), (1, -1), (1, 1), (2, 1)]
    rotation_index = 4
    reflection_index = 2


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
    rotation_index = 2
    reflection_index = 1


class L5(Polyomino):
    """
    o
    o
    o
    x o
    """

    name = "L"
    default_tiles = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    rotation_index = 4
    reflection_index = 2


class N5(Polyomino):
    """
      o
      o
    o o
    x
    """

    name = "N"
    default_tiles = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)]
    rotation_index = 4
    reflection_index = 2


class P5(Polyomino):
    """
    o o
    o o
    x
    """

    name = "P"
    default_tiles = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    rotation_index = 4
    reflection_index = 2


class T5(Polyomino):
    """
    o o o
      o
      x
    """

    name = "T"
    default_tiles = [(0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)]
    rotation_index = 4
    reflection_index = 1


class U5(Polyomino):
    """
    o   o
    x o o
    """

    name = "U"
    default_tiles = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]
    rotation_index = 4
    reflection_index = 1


class V5(Polyomino):
    """
        o
        o
    x o o
    """

    name = "V"
    default_tiles = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    rotation_index = 4
    reflection_index = 1


class W5(Polyomino):
    """
        o
      o o
    x o
    """

    name = "W"
    default_tiles = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
    rotation_index = 4
    reflection_index = 1


class X5(Polyomino):
    """
      o
    o o o
      x
    """

    name = "X"
    default_tiles = [(0, 0), (0, 1), (-1, 1), (1, 1), (0, 2)]
    rotation_index = 1
    reflection_index = 1


class Y5(Polyomino):
    """
      o
    o o
      o
      x
    """

    name = "Y"
    default_tiles = [(0, 0), (0, 1), (0, 2), (-1, 2), (0, 3)]
    rotation_index = 4
    reflection_index = 2


class Z5(Polyomino):
    """
    o o
      o
      x o
    """

    name = "Z"
    default_tiles = [(0, 0), (1, 0), (0, 1), (0, 2), (-1, 2)]
    rotation_index = 2
    reflection_index = 2


ALL_PENTOMINOES = [
    F5(),
    I5(),
    L5(),
    N5(),
    P5(),
    T5(),
    U5(),
    V5(),
    W5(),
    X5(),
    Y5(),
    Z5(),
]
