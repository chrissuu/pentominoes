import signal
from PolyominoSolver import PolyominoSolver
from Pentomino import ALL_PENTOMINOES
seconds = int

PUZZLE_NINE_INSTANCES = [
    # (3, 6,   (5, 5)),
    # (4, 12,  (7, 6)),
    # (5, 21,  (6, 8)),
    # (6, 34,  (9, 9)),
    # (7, 47,  (11, 9)),
    # (8, 62,  (11, 12)), # proves n=62 is upper bound (n=63 is UNSAT) in 620 seconds, bounded area by +3
    # (9, 78,  (11, 13)), # timed out after 1 hr on 79
    # (10, 93, (12, 14)), # ~ 36 minutes to solve LB=93, LB=94 might be UNSAT
    # (11, 103,(12, 15)), # LB=103 in 205seconds
    (12, 129,(20, 20)),
]

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run(name, width, height, inside_min, k, polyominoes, timeout: seconds, print_board=True):
    print(f"\n=== TEST CASE: {name} ===")
    solver = PolyominoSolver(
        width, 
        height, 
        inside_min,
        k, 
        polyominoes,
        True,
        False,
        model_save_path=f"models/{k}x{inside_min}x{width}x{height}.txt")

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

TIMEOUT = 60 * 60 * 24 * 14
WIDTH_EXPANDER = 0

for n, LB, area in PUZZLE_NINE_INSTANCES:
    width, height = area
    width = min(width + WIDTH_EXPANDER, 20)
    height = min(height + WIDTH_EXPANDER, 20)

    print(f"Validating LB={LB} for n={n}...")
    sat = run(
        f"Puzzle Nine {n}:{LB}",
        width, height,
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
