import itertools
from copy import deepcopy

from pysat.formula import CNF
from pysat.solvers import Glucose3

from polyomino import *


class PolyominoSolver:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.polyominoes = [
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

        self.var_counter = 1
        self.p_vars = {}
        self.cnf = CNF()

    def get_p_var(self, x: int, y: int, r: int, i: int) -> int:
        key = (x, y, r, i)
        if key not in self.p_vars:
            self.p_vars[key] = self.var_counter
            self.var_counter += 1
        return self.p_vars[key]

    def get_polyomino_tiles(
        self, polyomino_idx: int, x: int, y: int, rotation: int
    ) -> List[Tuple[int, int]]:
        polyomino = self.polyominoes[polyomino_idx]
        polyomino_copy: Polyomino = deepcopy(polyomino).rotate(rotation)
        tiles = [(x + tx, y + ty) for (tx, ty) in polyomino_copy.tiles]
        return tiles

    def is_valid_placement(self, tiles: List[Tuple[int, int]]) -> bool:
        return all(0 <= tx < self.width and 0 <= ty < self.height for tx, ty in tiles)

    def exactly_one_polyomino_constraint(self):
        for i in range(len(self.polyominoes)):
            placements = []
            for x in range(self.width):
                for y in range(self.height):
                    for r in range(4):
                        tiles = self.get_polyomino_tiles(i, x, y, r)
                        if self.is_valid_placement(tiles):
                            placements.append(self.get_p_var(x, y, r, i))

            self.cnf.append(placements)

            for p1, p2 in itertools.combinations(placements, 2):
                self.cnf.append([-p1, -p2])

    def add_no_overlap_constraints(self):
        for tx in range(self.width):
            for ty in range(self.height):
                occupiers = []

                for i in range(len(self.polyominoes)):
                    for x in range(self.width):
                        for y in range(self.height):
                            for r in range(4):
                                tiles = self.get_polyomino_tiles(i, x, y, r)
                                if self.is_valid_placement(tiles) and (tx, ty) in tiles:
                                    occupiers.append(self.get_p_var(x, y, r, i))

                for o1, o2 in itertools.combinations(occupiers, 2):
                    self.cnf.append([-o1, -o2])

    def solve(self) -> bool:
        self.exactly_one_polyomino_constraint()
        self.add_no_overlap_constraints()

        solver = Glucose3()
        for clause in self.cnf.clauses:
            solver.add_clause(clause)

        result = solver.solve()
        if result:
            self.model = solver.get_model()
            return True
        return False

    def get_placements(self) -> List[Tuple[int, int, int, int]]:
        placements = []
        for (x, y, r, i), var in self.p_vars.items():
            if var in self.model and self.model[var - 1] > 0:
                placements.append((x, y, r, i))
        return placements

    def get_board(self) -> List[List[str]]:
        placements = self.get_placements()
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        for x, y, r, i in placements:
            tiles = self.get_polyomino_tiles(i, x, y, r)
            polyomino_name = self.polyominoes[i].name
            for tx, ty in tiles:
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    grid[ty][tx] = polyomino_name

        return grid

    def print_board(self) -> None:
        grid = self.get_board()
        print("+" + "-" * self.width + "+")
        for row in reversed(grid):
            print("|" + "".join(row) + "|")
        print("+" + "-" * self.width + "+")


solver = PolyominoSolver(20, 20)

if solver.solve():
    print("Solution found!")
    solver.print_board()
