import signal
from PolyominoSolver import PolyominoSolver
from Pentomino import ALL_PENTOMINOES
seconds = int

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run(name, width, height, inside_min, k, polyominoes, timeout: seconds, print_board=True, save_path=None):
    print(f"\n=== TEST CASE: {name} ===")
    solver = PolyominoSolver(
        width, 
        height, 
        inside_min,
        k, 
        polyominoes,
        True,
        False,
        model_save_path=save_path)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        sat = solver.solve()
        signal.alarm(0)

        if sat:
            if print_board:
                solver.print_board()
            return True
        else:
            return False

    except TimeoutException:
        print(f"Timed out after {timeout} seconds on test case {name}.")
        return None

    finally:
        signal.alarm(0)
        print("=" * 40 + "\n")

run(name="20x20x128", 
    width=20, 
    height=20, 
    inside_min=128, 
    k=12, 
    polyominoes=ALL_PENTOMINOES, 
    timeout=60*60,
    print_board=True,
    save_path="models/12x128x20x20.txt")