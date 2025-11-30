import os
import re
import signal
import sys
from datetime import datetime
from math import sqrt, ceil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PolyominoSolver import PolyominoSolver

seconds = int

class TimeoutException(Exception):
    pass

def _resolve_log_dir(explicit: Optional[str]) -> Optional[str]:
    env_value = os.environ.get("PENTOMINO_LOG_DIR")
    if env_value is not None:
        env_value = env_value.strip()
        return env_value or None
    return explicit

def _resolve_log_elapsed(explicit: Optional[int]) -> Optional[int]:
    env_value = os.environ.get("PENTOMINO_LOG_ELAPSED_EVERY")
    if env_value is not None:
        try:
            parsed = int(env_value)
        except ValueError:
            return explicit
        return parsed if parsed > 0 else None
    return explicit


class RunLogger:
    """Context manager that mirrors stdout to a log file with filtering."""

    def __init__(
        self,
        name: str,
        run_parameters: Dict[str, Any],
        log_dir: Optional[Path],
        elapsed_log_every: Optional[int],
    ) -> None:
        self.name = name
        self.run_parameters = run_parameters
        self.log_dir = Path(log_dir) if log_dir else None
        self.elapsed_log_every = (
            elapsed_log_every if elapsed_log_every and elapsed_log_every > 0 else None
        )
        self.log_path: Optional[Path] = None
        self._log_file = None
        self._stdout = None
        self._buffer = ""
        self._elapsed_counter = 0
        self._enabled = self.log_dir is not None

    def __enter__(self) -> "RunLogger":
        if not self._enabled:
            return self

        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name).strip("_") or "run"
        self.log_path = self.log_dir / f"{timestamp}_{slug}.log"
        self._log_file = self.log_path.open("w", encoding="utf-8")
        self._write_header()

        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._enabled:
            return

        self.flush()

        if self._log_file:
            completion = datetime.now().isoformat()
            self._log_file.write(f"\n[run logger] Completed at {completion}\n")
            self._log_file.close()
            self._log_file = None

        if self._stdout:
            sys.stdout = self._stdout
            self._stdout = None

    def write(self, data: str) -> int:
        if self._stdout:
            self._stdout.write(data)
            self._stdout.flush()

        if not self._log_file:
            return len(data)

        normalized = data.replace("\r", "\n")
        self._buffer += normalized
        self._drain_buffer()
        return len(data)

    def flush(self) -> None:
        if not self._log_file:
            return
        self._drain_buffer(force=True)
        self._log_file.flush()

    def _write_header(self) -> None:
        if not self._log_file:
            return
        start_time = datetime.now().isoformat()
        self._log_file.write(f"Run '{self.name}' started at {start_time}\n")
        self._log_file.write("Parameters:\n")
        for key, value in self.run_parameters.items():
            self._log_file.write(f"  - {key}: {value}\n")
        self._log_file.write("-" * 60 + "\n")

    def _drain_buffer(self, force: bool = False) -> None:
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_line(line)
        if force and self._buffer:
            self._log_line(self._buffer)
            self._buffer = ""

    def _log_line(self, line: str) -> None:
        if not self._log_file:
            return
        if "[elapsed]" in line:
            self._elapsed_counter += 1
            if self.elapsed_log_every is None:
                return
            if self._elapsed_counter % self.elapsed_log_every != 0:
                return
        self._log_file.write(line.rstrip("\n") + "\n")

def timeout_handler(signum, frame):
    raise TimeoutException

def make_solver(k, inside_min, width, height, polyominoes,
                break_global_symmetries, break_polyomino_symmetries,
                model_save_path=None, formula_save_path=None):
    return PolyominoSolver(
        k=k,
        inside_tiles_minimum=inside_min,
        width=width,
        height=height,
        polyominoes=polyominoes,
        break_global_symmetries=break_global_symmetries,
        break_polyomino_symmetries=break_polyomino_symmetries,
        model_save_path=model_save_path,
        formula_save_path=formula_save_path,
    )

def solve_with_timeout(solver, timeout):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        sat = solver.solve()
        return sat
    except TimeoutException:
        return None
    finally:
        signal.alarm(0)

def run(
    name,
    k,
    inside_min,
    width,
    height,
    polyominoes: Iterable,
    break_global_symmetries,
    break_polyomino_symmetries,
    timeout: seconds,
    print_board=True,
    model_save_path=None,
    formula_save_path=None,
    log_dir: Optional[str] = "logs",
    log_elapsed_every: Optional[int] = 10,
):
    polyominoes = list(polyominoes)
    log_dir = _resolve_log_dir(log_dir)
    log_elapsed_every = _resolve_log_elapsed(log_elapsed_every)

    poly_names = [
        getattr(poly, "name", None) or poly.__class__.__name__ for poly in polyominoes
    ]
    log_params: Dict[str, Any] = {
        "k": k,
        "inside_min": inside_min,
        "width": width,
        "height": height,
        "polyominoes": ", ".join(poly_names),
        "break_global_symmetries": break_global_symmetries,
        "break_polyomino_symmetries": break_polyomino_symmetries,
        "timeout_seconds": timeout,
        "print_board": print_board,
        "model_save_path": model_save_path or "None",
        "formula_save_path": formula_save_path or "None",
        "log_dir": log_dir or "disabled",
        "log_elapsed_every": log_elapsed_every if log_elapsed_every else "disabled",
    }
    logger = RunLogger(
        name=name,
        run_parameters=log_params,
        log_dir=Path(log_dir) if log_dir else None,
        elapsed_log_every=log_elapsed_every,
    )

    result = None
    with logger:
        print(f"\n=== TEST CASE: {name} ===")

        solver = make_solver(
            k=k,
            inside_min=inside_min,
            width=width,
            height=height,
            polyominoes=polyominoes,
            break_global_symmetries=break_global_symmetries,
            break_polyomino_symmetries=break_polyomino_symmetries,
            model_save_path=model_save_path,
            formula_save_path=formula_save_path,
        )

        sat = solve_with_timeout(solver, timeout)

        if sat is None:
            print(f"Timed out after {timeout} seconds on {name}.")
            print("=" * 40 + "\n")
            result = None
        elif sat:
            if print_board:
                solver.print_board()
            print("=" * 40 + "\n")
            result = True
        else:
            print("=" * 40 + "\n")
            result = False

    return result

def validate(
    k,
    inside_min,
    width,
    height,
    polyominoes,
    break_global_symmetries,
    break_polyomino_symmetries,
    model_save_path,
    formula_save_path,
):
    solver = make_solver(
        k=k,
        inside_min=inside_min,
        width=width,
        height=height,
        polyominoes=polyominoes,
        break_global_symmetries=break_global_symmetries,
        break_polyomino_symmetries=break_polyomino_symmetries,
        model_save_path=model_save_path,
        formula_save_path=formula_save_path,
    )

    print("Loading model...")
    solver.load_model(model_save_path)

    print("Loading formula...")
    solver.load_formula(formula_save_path)

    for clause in solver.cnf.clauses:
        if not any(
            (abs(lit) <= len(solver.model) and solver.model[abs(lit) - 1] == lit)
            for lit in clause
        ):
            print(f"Model violates clause: {clause}")
            return False

    print("All CNF clauses satisfied.")

    print("Rebuilding constraints for structural validation...")
    model_backup = list(solver.model) if solver.model is not None else None
    solver.reset_encoding_state()
    solver.build_constraints()
    if model_backup is not None:
        solver.model = model_backup

    try:
        solver.validate_model()
        print("Structural validation passed.")
        return True
    except AssertionError as e:
        print("Model failed structural validation:")
        print(e)
        solver.print_board()
        return False

def w(k, l, q):
    """ Lower bound width formula """
    return ceil(1 + k*l/4 + sqrt((k*l)**2 / 16 - k*l/2 - q + 1))
