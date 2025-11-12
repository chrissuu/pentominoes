import itertools
import time
import argparse
from copy import deepcopy

from pysat.formula import CNF, IDPool
from pysat.solvers import Cadical195
from pysat.card import CardEnc, EncType

from typing import List, Tuple, Dict

from polyomino import *
from Pentomino import *
from Tetromino import *

import functools

def log_diff(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        prev = len(self.cnf.clauses)
        computation = fn(self, *args, **kwargs)
        after = len(self.cnf.clauses)
        print(f"Constraint {fn.__name__} added {after - prev} clauses.")
        return computation
    
    return wrapper
    

class PolyominoSolver:
    def __init__(self, width: int, height: int, inside_tiles_minimum, polyominoes):
        self.width = width
        self.height = height
        self.inside_tiles_minimum = inside_tiles_minimum
        self.polyominoes = polyominoes

        self.var_counter = 1

        self.p_vars: Dict[Tuple[int, int, int, int], int] = {}

        self.tf_vars: Dict[Tuple[int, int], int] = {}
        self.ti_vars: Dict[Tuple[int, int], int] = {}
        self.to_vars: Dict[Tuple[int, int], int] = {}

        self.cnf = CNF()
        self.model = None

    def _new_var(self) -> int:
        v = self.var_counter
        self.var_counter += 1
        return v

    def get_p_var(self, x: int, y: int, r: int, i: int) -> int:
        """
        Polyomino tile
        
        See Polyomino.py for an encoding
        """
        key = (x, y, r, i)
        if key not in self.p_vars:
            self.p_vars[key] = self._new_var()
        return self.p_vars[key]

    def get_tf_var(self, x: int, y: int) -> int:
        """
        Fence tiles

        Returns the variable representing whether (x,y) is
        a fence tile. creates one if it does not exist.
        """
        key = (x, y)
        if key not in self.tf_vars:
            self.tf_vars[key] = self._new_var()
        return self.tf_vars[key]

    def get_ti_var(self, x: int, y: int) -> int:
        """
        Inside tiles

        Returns the variable representing whether (x,y) is
        an inside tile. creates one if it does not exist.
        """
        key = (x, y)
        if key not in self.ti_vars:
            self.ti_vars[key] = self._new_var()
        return self.ti_vars[key]

    def get_to_var(self, x: int, y: int) -> int:
        """
        Outside tiles

        Returns the variable representing whether (x,y) is
        an outside tile. Creates one if it does not exist.
        """
        key = (x, y)
        if key not in self.to_vars:
            self.to_vars[key] = self._new_var()
        return self.to_vars[key]

    def get_polyomino_tiles(
        self, polyomino_idx: int, x: int, y: int, rotation: int
    ) -> List[Tuple[int, int]]:
        polyomino = self.polyominoes[polyomino_idx]
        polyomino_copy: Polyomino = deepcopy(polyomino)
        polyomino_copy = polyomino_copy.rotate(rotation)
        tiles = [(x + tx, y + ty) for (tx, ty) in polyomino_copy.tiles]
        return tiles

    def is_valid_placement(self, tiles: List[Tuple[int, int]]) -> bool:
        return all(0 <= tx < self.width and 0 <= ty < self.height for tx, ty in tiles)

    def neighbors4(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Computes the neighboring coordinates, directly adjacent to
        (x,y). Filters to ensure that coordinates fall within bounds.
        """
        candidates = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        return [(nx, ny) for (nx, ny) in candidates
                if 0 <= nx < self.width and 0 <= ny < self.height]

    def neighbors8(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Computes the neighboring coordinates, directly adjacent 
        and diagonal to (x,y). Filters to ensure that coordinates 
        fall within bounds.
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        candidates = [(x + dx, y + dy) for (dx, dy) in dirs]
        return [(nx, ny) for (nx, ny) in candidates
                if 0<= nx < self.width and 0 <= ny < self.height]
    
    def build_map(self):
        cell_to_placements: Dict[Tuple[int, int], List[int]] = {
            (x, y): [] for x in range(self.width) for y in range(self.height)
        }

        for i, _ in enumerate(self.polyominoes):
            for x in range(self.width):
                for y in range(self.height):
                    for r in range(4):
                        tiles = self.get_polyomino_tiles(i, x, y, r)
                        if not self.is_valid_placement(tiles):
                            continue
                        p_var = self.get_p_var(x, y, r, i)
                        for (tx, ty) in tiles:
                            cell_to_placements[(tx, ty)].append(p_var)
        self.cell_to_placements = cell_to_placements

# BEGIN CONSTRAINTS DEFINITIONS

    @log_diff
    def add_exactly_one_polyomino_constraint(self):
        """
        For each polyomino i, choose exactly one placement (corner x,y and rotation r).
        Uses a single cardinality encoding instead of pairwise exclusions.
        """
        vpool = IDPool(start_from=self.var_counter)
        for i in range(len(self.polyominoes)):
            placements = []
            for x in range(self.width):
                for y in range(self.height):
                    for r in range(4):
                        tiles = self.get_polyomino_tiles(i, x, y, r)
                        if self.is_valid_placement(tiles):
                            placements.append(self.get_p_var(x, y, r, i))

            if not placements:
                continue

            enc_eq = CardEnc.equals(lits=placements, bound=1,
                                    encoding=EncType.seqcounter, vpool=vpool)
            self.cnf.extend(enc_eq.clauses)

        if vpool.top is not None:
            self.var_counter = max(self.var_counter, vpool.top + 1)
    
    @log_diff
    def add_no_overlap_constraints(self):
        """
        Each tile is covered by at most one polyomino.
        """
        if not self.cell_to_placements:
            self.build_map()

        vpool = IDPool(start_from=self.var_counter)
        for occupiers in self.cell_to_placements.values():
            if len(occupiers) > 1:
                enc_atmost = CardEnc.atmost(lits=occupiers, bound=1,
                                            encoding=EncType.seqcounter, vpool=vpool)
                self.cnf.extend(enc_atmost.clauses)

        if vpool.top is not None:
            self.var_counter = max(self.var_counter, vpool.top + 1)

    @log_diff
    def add_tile_partition_constraints(self):
        for x in range(self.width):
            for y in range(self.height):
                cf = self.get_tf_var(x, y)
                ci = self.get_ti_var(x, y)
                co = self.get_to_var(x, y)

                self.cnf.append([cf, ci, co])

                self.cnf.append([-cf, -ci])
                self.cnf.append([-cf, -co])
                self.cnf.append([-ci, -co])

    @log_diff
    def add_link_fence_to_placements_constraint(self):
        if not self.cell_to_placements:
            self.build_map()

        for (tx, ty), occupiers in self.cell_to_placements.items():
            cf = self.get_tf_var(tx, ty)

            if occupiers:
                for p in occupiers:
                    self.cnf.append([-p, cf])
                self.cnf.append([-cf] + occupiers)
            else:
                self.cnf.append([-cf])

    @log_diff
    def add_outside_adjacency_constraints(self):
        """
        Encodes the constraint that adjacent tiles
        to an outside tile (including diagonals) should be 
        either an outside tile or a fence (polyomino)
        """
        for x in range(self.width):
            for y in range(self.height):
                to_xy = self.get_to_var(x, y)
                for nx, ny in self.neighbors8(x, y):
                    ti_n = self.get_ti_var(nx, ny)
                    self.cnf.append([-to_xy, -ti_n])

    @log_diff
    def add_cardinality_constraints(self):
        """
        Ensures that there are at least self.inside_tiles_minimum
        inside tiles.

        Also ensures that there are exactly total_tiles tiles.
        This ensures that pentominos do not overlap.
        """
        tf_literals = [self.get_tf_var(x, y) for x in range(self.width) for y in range(self.height)]
        ti_literals = [self.get_ti_var(x, y) for x in range(self.width) for y in range(self.height)]

        total_tiles = len(self.polyominoes[0].default_tiles) * len(self.polyominoes)

        vpool = IDPool(start_from=self.var_counter)

        enc_eq = CardEnc.equals(lits=tf_literals, bound=total_tiles, encoding=EncType.seqcounter, vpool=vpool)
        self.cnf.extend(enc_eq.clauses)

        enc_ge = CardEnc.atleast(lits=ti_literals, bound=self.inside_tiles_minimum, encoding=EncType.seqcounter, vpool=vpool)
        self.cnf.extend(enc_ge.clauses)

        if vpool.top is not None:
            self.var_counter = max(self.var_counter, vpool.top + 1)

    @log_diff
    def add_outside_border_constraints(self):
        """
        Force every boundary (perimeter) tile to be outside
        """
        if self.width == 0 or self.height == 0:
            return

        for x in range(self.width):
            self.cnf.append([self.get_to_var(x, 0)])
            self.cnf.append([self.get_to_var(x, self.height - 1)])

        for y in range(self.height):
            self.cnf.append([self.get_to_var(0, y)])             
            self.cnf.append([self.get_to_var(self.width - 1, y)])

# END CONSTRAINTS DEFINITIONS

    def solve(self) -> bool:
        print("BEGIN BUILDING MAP")
        t0 = time.time()
        self.build_map()
        
        build_map_time = time.time() - t0
        print(f"Built map in {build_map_time:.2f} seconds")
        print()
        print("BEGIN CONSTRAINT BUILDING")
        t1 = time.time()

        # TODO: write down how many clauses each constraint
        # contributes to the encoding
        self.add_exactly_one_polyomino_constraint()
        self.add_no_overlap_constraints()
        self.add_tile_partition_constraints()
        self.add_link_fence_to_placements_constraint()
        self.add_outside_adjacency_constraints()
        self.add_cardinality_constraints()
        self.add_outside_border_constraints()

        build_time = time.time() - t1

        num_clauses = len(self.cnf.clauses)
        num_vars = self.var_counter - 1
        avg_clause_len = (
            sum(len(c) for c in self.cnf.clauses) / num_clauses if num_clauses else 0
        )

        print(f"CNF built in {build_time:.2f} seconds.")
        print(f"  Variables: {num_vars}")
        print(f"  Clauses:   {num_clauses}")
        print(f"  Avg clause length: {avg_clause_len:.2f}")
        print()

        print("BEGIN SOLVING")
        solver = Cadical195()
        for clause in self.cnf.clauses:
            solver.add_clause(clause)

        t2 = time.time()
        result = solver.solve()
        solve_time = time.time() - t2

        print(f"Solving finished in {solve_time:.2f} seconds.")
        print(f"SAT result: {'SATISFIABLE' if result else 'UNSATISFIABLE'}")

        if result:
            self.model = solver.get_model()
            return True
        return False

    def _is_true(self, var: int) -> bool:
        return self.model[var - 1] > 0

    def get_placements(self) -> List[Tuple[int, int, int, int]]:
        placements = []
        for (x, y, r, i), var in self.p_vars.items():
            if self._is_true(var):
                placements.append((x, y, r, i))
        return placements

    def get_board(self) -> List[List[str]]:
        """
        Inside tiles are "+", outside tiles are "-"
        """
        grid = [["-" for _ in range(self.width)] for _ in range(self.height)]

        for x in range(self.width):
            for y in range(self.height):
                ci = self.get_ti_var(x, y)
                if self._is_true(ci):
                    grid[y][x] = "+"

        for (x, y, r, i) in self.get_placements():
            tiles = self.get_polyomino_tiles(i, x, y, r)
            name = self.polyominoes[i].name
            for tx, ty in tiles:
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    grid[ty][tx] = name

        return grid

    def print_board(self) -> None:
        grid = self.get_board()
        print("*" + "*" * self.width + "*")
        for row in reversed(grid):
            print("*" + "".join(row) + "*")
        print("*" + "*" * self.width + "*")
