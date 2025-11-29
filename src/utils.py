import signal
from math import sqrt, ceil
from .PolyominoSolver import PolyominoSolver

seconds = int

class TimeoutException(Exception):
    pass

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
    polyominoes,
    break_global_symmetries,
    break_polyomino_symmetries,
    timeout: seconds,
    print_board=True,
    model_save_path=None,
    formula_save_path=None,
):
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
        return None

    if sat:
        if print_board:
            solver.print_board()
        print("=" * 40 + "\n")
        return True

    print("=" * 40 + "\n")
    return False

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
