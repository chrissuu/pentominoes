from PolyominoSolver import PolyominoSolver

from Pentomino import *
from Tetromino import *
"""
Note on notation:

We use to, ti, tf to represent:
-> "Tile outside"
-> "Tile inside"
-> "Tile fence"
"""
def solver():
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


# ALL_TETROMINOES = [
#     I4(),
#     O4(),
#     T4(),
#     L4(),
#     N4(),
# ]

if __name__ == "__main__":
    import copy
    solver = PolyominoSolver(10, 10, 9, ALL_TETROMINOES)
    assert solver.solve()

    solver = PolyominoSolver(16, 16, 80, ALL_TETROMINOES + copy.deepcopy(ALL_TETROMINOES))
    if solver.solve():
        print("Solution found!")
        solver.print_board()
    else:
        print("No solution.")