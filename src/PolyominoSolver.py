import functools
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from itertools import product
from typing import Dict, Sequence

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool

from polyomino import *


def log_diff(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        prev = len(self.cnf.clauses)
        computation = fn(self, *args, **kwargs)
        after = len(self.cnf.clauses)
        print(f"Constraint {fn.__name__} added {after - prev} clauses.")
        return computation

    return wrapper


def solve_in_subprocess(clauses):
    """
    Standalone helper for multiprocessing: takes a list of CNF clauses (list[list[int]])
    and returns (result, model), where result is a bool and model is a list[int] or None.
    """
    from pysat.solvers import Cadical195

    solver = Cadical195()
    for c in clauses:
        solver.add_clause(c)

    result = solver.solve()
    model = solver.get_model() if result else None
    return result, model


class PolyominoSolver:
    def __init__(
        self,
        width: int,
        height: int,
        inside_tiles_minimum: int,
        k: int | None,
        polyominoes: Sequence[Polyomino],
        break_global_symmetries: bool = True,
        break_polyomino_symmetries: bool = True,
        use_sbva: bool = False,
        model_save_path=None,
    ):
        self.width = width
        self.height = height
        self.inside_tiles_minimum = inside_tiles_minimum
        self.polyominoes = deepcopy(polyominoes)  # Make copy to avoid modifying input
        self.k = k
        self.model_save_path = model_save_path

        self.use_sbva = use_sbva
        self.break_global_symmetries = break_global_symmetries
        if self.break_global_symmetries:
            # Symmetry breaking: fix one polyomino to a canonical rotation
            p_fixed_rotation = next(
                p for p in self.polyominoes if p.rotation_index == 4
            )
            p_fixed_rotation.rotation_index = 1
            # Symmetry breaking: fix one polyomino to a canonical reflection
            p_fixed_reflection = next(
                p for p in self.polyominoes if p.reflection_index == 2
            )
            p_fixed_reflection.reflection_index = 1

        self.break_polyomino_symmetries = break_polyomino_symmetries
        if not self.break_polyomino_symmetries:
            # Try all rotations and reflections for all polyominoes, regardless of symmetry
            for poly in self.polyominoes:
                poly.rotation_index = 4
                poly.reflection_index = 2

        self.vpool = IDPool()
        self.p_vars = set()
        self.cell_to_placements = None
        self.use_vars = {}

        self.cnf = CNF()
        self.model = None

    def get_p_var(self, x: int, y: int, r: int, m: int, i: int) -> int:
        """
        Polyomino tile

        Represents placing polyomino i at position (x,y) with rotation r and reflection m.
        See Polyomino.py for an encoding
        """
        self.p_vars.add((x, y, r, m, i))
        return self.vpool.id(f"p_{x}_{y}_{r}_{m}_{i}")

    def get_tf_var(self, x: int, y: int) -> int:
        """
        Fence tiles

        Returns the variable representing whether (x,y) is
        a fence tile. creates one if it does not exist.
        """
        return self.vpool.id(f"tf_{x}_{y}")

    def get_ti_var(self, x: int, y: int) -> int:
        """
        Inside tiles

        Returns the variable representing whether (x,y) is
        an inside tile. creates one if it does not exist.
        """
        return self.vpool.id(f"ti_{x}_{y}")

    def get_to_var(self, x: int, y: int) -> int:
        """
        Outside tiles

        Returns the variable representing whether (x,y) is
        an outside tile. Creates one if it does not exist.
        """
        return self.vpool.id(f"to_{x}_{y}")

    def get_use_var(self, i: int) -> int:
        return self.vpool.id(f"use_{i}")

    def get_polyomino_tiles(
        self, polyomino_idx: int, x: int, y: int, rotation: int, reflect: int
    ) -> List[Tuple[int, int]]:
        if reflect != 0 and reflect != 1:
            raise ValueError("reflect value must be 0 or 1")

        polyomino = self.polyominoes[polyomino_idx]
        polyomino_copy = deepcopy(polyomino).rotate(rotation)
        if reflect != 0:
            polyomino_copy.reflect_horizontally()
        polyomino_copy.recenter_at_origin()
        tiles = [(x + tx, y + ty) for (tx, ty) in polyomino_copy.tiles]
        return tiles

    def is_valid_placement(self, tiles: List[Tuple[int, int]]) -> bool:
        """
        Checks if all tiles fall within the valid area of the board.
        Border cells should not be occupied by polyominoes.
        """
        return all(
            1 <= tx < self.width - 1 and 1 <= ty < self.height - 1 for tx, ty in tiles
        )

    def neighbors4(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Computes the neighboring coordinates, directly adjacent to
        (x,y). Filters to ensure that coordinates fall within bounds.
        """
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [
            (nx, ny)
            for (nx, ny) in candidates
            if 0 <= nx < self.width and 0 <= ny < self.height
        ]

    def neighbors8(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Computes the neighboring coordinates, directly adjacent
        and diagonal to (x,y). Filters to ensure that coordinates
        fall within bounds.
        """
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        candidates = [(x + dx, y + dy) for (dx, dy) in dirs]
        return [
            (nx, ny)
            for (nx, ny) in candidates
            if 0 <= nx < self.width and 0 <= ny < self.height
        ]

    def build_map(self):
        cell_to_placements: Dict[Tuple[int, int], List[int]] = {
            (x, y): [] for x in range(self.width) for y in range(self.height)
        }

        for i, polyomino in enumerate(self.polyominoes):
            for x, y, r, m in product(
                range(self.width),
                range(self.height),
                range(polyomino.rotation_index),
                range(polyomino.reflection_index),
            ):
                tiles = self.get_polyomino_tiles(i, x, y, r, m)
                if not self.is_valid_placement(tiles):
                    continue
                p_var = self.get_p_var(x, y, r, m, i)
                for tx, ty in tiles:
                    cell_to_placements[(tx, ty)].append(p_var)
        self.cell_to_placements = cell_to_placements

    # BEGIN CONSTRAINTS DEFINITIONS

    @log_diff
    def add_exactly_one_polyomino_constraint(self):
        """
        For each polyomino i, choose exactly one placement (corner x,y and rotation r).
        Uses a single cardinality encoding instead of pairwise exclusions.
        """
        for i, polyomino in enumerate(self.polyominoes):
            placements = []
            for x, y, r, m in product(
                range(self.width),
                range(self.height),
                range(polyomino.rotation_index),
                range(polyomino.reflection_index),
            ):
                tiles = self.get_polyomino_tiles(i, x, y, r, m)
                if self.is_valid_placement(tiles):
                    placements.append(self.get_p_var(x, y, r, m, i))

            if not placements:
                continue

            enc_eq = CardEnc.equals(
                lits=placements, bound=1, encoding=EncType.seqcounter, vpool=self.vpool
            )
            self.cnf.extend(enc_eq.clauses)

    @log_diff
    def add_no_overlap_constraints(self):
        """
        Each tile is covered by at most one polyomino.
        """
        if not self.cell_to_placements:
            self.build_map()

        for occupiers in self.cell_to_placements.values():
            if len(occupiers) > 1:
                enc_atmost = CardEnc.atmost(
                    lits=occupiers,
                    bound=1,
                    encoding=EncType.seqcounter,
                    vpool=self.vpool,
                )
                self.cnf.extend(enc_atmost.clauses)

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
    def add_total_tiles_constraint(self):
        """
        Ensures that the total number of filled tiles equals the total
        number of polyomino tiles combined. This prevents overlaps.
        """
        tf_literals = [
            self.get_tf_var(x, y) for x in range(self.width) for y in range(self.height)
        ]

        if self.k is not None:
            total_tiles = len(self.polyominoes[0].default_tiles) * self.k
        else:
            total_tiles = len(self.polyominoes[0].default_tiles) * len(self.polyominoes)

        enc_eq = CardEnc.equals(
            lits=tf_literals,
            bound=total_tiles,
            encoding=EncType.seqcounter,
            vpool=self.vpool,
        )
        self.cnf.extend(enc_eq.clauses)

    @log_diff
    def add_inside_tiles_constraint(self):
        """
        Ensures that there are at least `self.inside_tiles_minimum`
        inside tiles.
        """
        ti_literals = [
            self.get_ti_var(x, y) for x in range(self.width) for y in range(self.height)
        ]

        enc_ge = CardEnc.atleast(
            lits=ti_literals,
            bound=self.inside_tiles_minimum,
            encoding=EncType.seqcounter,
            vpool=self.vpool,
        )
        self.cnf.extend(enc_ge.clauses)

    def add_cardinality_constraints(self):
        self.add_total_tiles_constraint()
        self.add_inside_tiles_constraint()

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

    @log_diff
    def add_polyomino_selection_constraint(self, c):
        use_vars = []
        all_placements_per_poly = []

        for i, polyomino in enumerate(self.polyominoes):
            ui = self.get_use_var(i)
            use_vars.append(ui)

            placements = []
            for x, y, r, m in product(
                range(self.width),
                range(self.height),
                range(polyomino.rotation_index),
                range(polyomino.reflection_index),
            ):
                tiles = self.get_polyomino_tiles(i, x, y, r, m)
                if self.is_valid_placement(tiles):
                    placements.append(self.get_p_var(x, y, r, m, i))

            all_placements_per_poly.append(placements)

        for ui, placements in zip(use_vars, all_placements_per_poly):
            if not placements:
                self.cnf.append([-ui])
                continue

            amo = CardEnc.atmost(
                lits=placements, bound=1, encoding=EncType.seqcounter, vpool=self.vpool
            )
            self.cnf.extend(amo.clauses)

            self.cnf.append([-ui] + placements)

            for p in placements:
                self.cnf.append([-p, ui])

        enc_eq = CardEnc.equals(
            lits=use_vars, bound=c, encoding=EncType.seqcounter, vpool=self.vpool
        )
        self.cnf.extend(enc_eq.clauses)

    @log_diff
    def add_global_symmetry_breaking_constraints(self):
        """
        Break translation symmetry by forcing solution to have at least one tile in the first row and column
        """
        self.cnf.append([self.get_tf_var(x, 1) for x in range(1, self.width - 1)])
        self.cnf.append([self.get_tf_var(1, y) for y in range(1, self.height - 1)])

    @log_diff
    def add_fence_touch_inside_constraint(self):
        """
        Enforce that:
          (1) there exists at least one fence cell that is 4-neighbor-adjacent
              to an inside cell, and
          (2) there exists at least one fence cell that is 4-neighbor-adjacent
              to an outside cell.

        We do this by introducing auxiliary variables that represent
        "this particular fence–inside (or fence–outside) pair is active",
        then require that at least one such variable is true in each group.
        """
        inside_touch_lits = []
        outside_touch_lits = []

        for x in range(self.width):
            for y in range(self.height):
                tf = self.get_tf_var(x, y)
                for nx, ny in self.neighbors4(x, y):
                    ti = self.get_ti_var(nx, ny)
                    to = self.get_to_var(nx, ny)

                    v_in = self.vpool.id(f"touch_in_{x}_{y}_{nx}_{ny}")
                    self.cnf.append([-v_in, tf])
                    self.cnf.append([-v_in, ti])
                    inside_touch_lits.append(v_in)

                    v_out = self.vpool.id(f"touch_out_{x}_{y}_{nx}_{ny}")
                    self.cnf.append([-v_out, tf])
                    self.cnf.append([-v_out, to])
                    outside_touch_lits.append(v_out)

        if inside_touch_lits:
            self.cnf.append(inside_touch_lits)

        if outside_touch_lits:
            self.cnf.append(outside_touch_lits)

    def build_constraints(self):
        print("BEGIN BUILDING MAP")
        t0 = time.time()
        self.build_map()

        build_map_time = time.time() - t0
        print(f"Built map in {build_map_time:.2f} seconds")
        print()
        print("BEGIN CONSTRAINT BUILDING")
        t1 = time.time()

        if self.k is not None:
            self.add_polyomino_selection_constraint(self.k)
        else:
            self.add_exactly_one_polyomino_constraint()
        self.add_cardinality_constraints()
        # self.add_no_overlap_constraints()
        self.add_tile_partition_constraints()
        self.add_link_fence_to_placements_constraint()
        self.add_outside_adjacency_constraints()
        self.add_outside_border_constraints()
        self.add_fence_touch_inside_constraint()

        if self.break_global_symmetries:
            self.add_global_symmetry_breaking_constraints()

        build_time = time.time() - t1

        num_clauses = len(self.cnf.clauses)
        num_vars = self.vpool.top
        avg_clause_len = (
            sum(len(c) for c in self.cnf.clauses) / num_clauses if num_clauses else 0
        )

        if self.use_sbva:
            print("SIMPLIFYING CNF WITH SBVA...")
            t2 = time.time()
            self.simplify_cnf_with_sbva()
            sbva_time = time.time() - t2
            print(f"SBVA simplification completed in {sbva_time:.2f} seconds.")
            print()

        print(f"CNF built in {build_time:.2f} seconds.")
        print(f"  Variables: {num_vars}")
        print(f"  Clauses:   {num_clauses}")
        print(f"  Avg clause length: {avg_clause_len:.2f}")
        print()

    def simplify_cnf_with_sbva(self):
        with tempfile.NamedTemporaryFile(mode="w") as temp_in:
            with tempfile.NamedTemporaryFile(mode="r") as temp_out:
                self.cnf.to_file(temp_in.name)
                sbva_path = os.path.join(
                    os.path.dirname(__file__), "..", "SBVA", "sbva"
                )
                subprocess.run(
                    [sbva_path, "-i", temp_in.name, "-o", temp_out.name], check=True
                )
                simplified_cnf = CNF(from_file=temp_out.name)
                self.cnf = simplified_cnf

    # END CONSTRAINTS DEFINITIONS

    def solve(self) -> bool:
        import multiprocessing as mp
        import time

        self.build_constraints()
        print("BEGIN SOLVING")

        clauses = self.cnf.clauses

        with mp.Pool(1) as pool:
            worker = pool._pool[0]
            self._solver_pid = worker.pid

            self._start_progress_timer(interval=0.1)

            async_res = pool.apply_async(solve_in_subprocess, (clauses,))

            t0 = time.time()
            result, model = async_res.get()
            solve_time = time.time() - t0

        self._stop_progress_timer()
        time.sleep(0.05)

        print(f"\rSolving finished in {self._format_elapsed(solve_time)}.")
        print(f"SAT result: {'SATISFIABLE' if result else 'UNSATISFIABLE'}")

        if result:
            self.model = model
            if self.model_save_path is not None:
                self.save_model(self.model_save_path)
            return True

        return False

    def _format_elapsed(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        seconds = seconds % 60
        if minutes < 60:
            return f"{minutes}m {seconds:.1f}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"

    def _start_progress_timer(self, interval: float = 0.1):
        import sys, time, threading, psutil

        self._running_solve = True
        start = time.time()

        child = psutil.Process(self._solver_pid)

        def loop():
            while self._running_solve:
                try:
                    elapsed = time.time() - start

                    cpu = child.cpu_percent(interval=None)
                    mem = child.memory_info().rss / 1e6

                    msg = (
                        f"[elapsed] {self._format_elapsed(elapsed)} "
                        f"| CPU {cpu:5.1f}% "
                        f"| RAM {mem:7.1f} MB"
                    )

                    sys.stdout.write("\r" + msg + " " * 10)
                    sys.stdout.flush()

                    time.sleep(interval)

                except Exception:
                    break

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._progress_thread = t

    def _stop_progress_timer(self):
        self._running_solve = False
        print("\r", end="", flush=True)

    def _is_true(self, var: int) -> bool:
        return self.model[var - 1] > 0

    def get_placements(self) -> List[Tuple[int, int, int, int, int]]:
        placements = []
        for p_var in self.p_vars:
            if self._is_true(self.get_p_var(*p_var)):
                placements.append(p_var)
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

        for x, y, r, m, i in self.get_placements():
            tiles = self.get_polyomino_tiles(i, x, y, r, m)
            name = self.polyominoes[i].name
            for tx, ty in tiles:
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    grid[ty][tx] = name

        return grid

    def print_board(self, solution_file: str | None = None) -> None:
        """
        If solution_file is provided, loads that model and displays the board for it.
        Otherwise uses self.model (i.e., the live solver result).
        """
        old_model = self.model

        if solution_file is not None:
            loaded = self._load_model_file(solution_file)
            if not loaded:
                raise ValueError(f"Could not load a valid model from {solution_file}")
            self.model = loaded

        grid = self.get_board()

        print("*" + "*" * self.width + "*")
        for row in reversed(grid):
            print("*" + "".join(row) + "*")
        print("*" + "*" * self.width + "*")

        self.model = old_model

    def save_to(self, file):
        self.cnf.to_file(file)

    def _load_model_file(self, file_path: str) -> list[int]:
        """
        Load a model file saved by save_model().
        Expected format:
            c ...
            s SATISFIABLE
            v lit1 lit2 ... 0
            ...
        Returns a flat list of ints (positive or negative).
        """
        lits = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("c") or line.startswith("s"):
                    continue
                if line.startswith("v "):
                    parts = line.split()[1:]
                    for tok in parts:
                        if tok == "0":
                            break
                        lits.append(int(tok))
        return lits

    def save_model(self, file_path: str) -> None:
        """
        Save the current model (variable assignment) to `file_path`
        in a standard SAT solver output format:

            c comment
            s SATISFIABLE
            v lit1 lit2 ... 0
            v lit_k ... 0

        where each `lit` is a signed integer corresponding to a variable.
        """
        with open(file_path, "w") as f:
            f.write("c Model generated by PolyominoSolver\n")
            f.write("s SATISFIABLE\n")

            chunk_size = 20
            for i in range(0, len(self.model), chunk_size):
                chunk = self.model[i : i + chunk_size]
                f.write("v " + " ".join(str(lit) for lit in chunk) + " 0\n")
