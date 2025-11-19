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

# ALL_TETROMINOES = [
#     I4(),
#     O4(),
#     T4(),
#     L4(),
#     N4(),
# ]

if __name__ == "__main__":
    import copy
    solver = PolyominoSolver(18, 18, 129, None, ALL_PENTOMINOES)
    # solver.build_constraints()
    # solver.save_to("128.cnf")
    solver.solve()
