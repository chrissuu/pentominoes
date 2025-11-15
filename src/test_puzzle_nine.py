import signal
from PolyominoSolver import PolyominoSolver
from Pentomino import ALL_PENTOMINOES
seconds = int

PUZZLE_NINE_INSTANCES = [
    (3, 6,   (5, 5)),
    (4, 12,  (7, 6)),
    (5, 21,  (6, 8)),
    (6, 32,  (9, 9)),
    (7, 43,  (11, 9)),
    (8, 61,  (11, 12)),
    (9, 70,  (11, 13)),
    (10, 84, (12, 14)),
    (11, 102,(12, 15)),
    (12, 128,(18, 16)),
]

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run(name, width, height, inside_min, k, polyominoes, timeout: seconds, print_board=True):
    print(f"\n=== TEST CASE: {name} ===")
    solver = PolyominoSolver(width, height, inside_min, k, polyominoes)

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

results = {}

TIMEOUT = 60

for n, LB, area in PUZZLE_NINE_INSTANCES:
    width, height = area
    print(f"Validating LB={LB} for n={n}...")
    sat = run(
        f"Puzzle Nine {n}:{LB}",
        width + 3, height + 3,
        LB, n,
        ALL_PENTOMINOES,
        TIMEOUT,
        print_board=True
    )

    if sat is None:
        print(f"Validation timed out for n={n}. Skipping.")
        results[n] = f"{LB} -> t"
        continue

    if not sat:
        print(f"ERROR: Provided LB {LB} is UNSAT for n={n}.")
        results[n] = f"{LB} -> UNSAT"
        continue

    curr = LB + 1
    while True:
        print(f"Testing n={n} LB={curr}...")
        sat = run(
            f"Puzzle Nine {n}:{curr}",
            width + 3, height + 3,
            curr, n,
            ALL_PENTOMINOES,
            TIMEOUT,
            print_board=False
        )

        if sat is None:
            print(f"Timeout at LB={curr} for n={n}.")
            results[n] = f"{LB} -> t" if curr == LB + 1 else f"{LB} -> {curr-1} (t)"
            break

        if sat is False:
            print(f"UNSAT at LB={curr} for n={n}.")
            results[n] = f"{LB} -> {curr}"
            break

        curr += 1


print("\n====== SUMMARY ======")
for n, summary in results.items():
    print(f"{n}: {summary}")
