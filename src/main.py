import signal
from math import sqrt, ceil
from PolyominoSolver import PolyominoSolver
from Pentomino import ALL_PENTOMINOES
from utils import *

seconds = int

PUZZLE_NINE_INSTANCES = [
    (3, 6),
    # (4, 12),
    # (5, 21),
    # (6, 34),
    # (7, 47),
    # (8, 62),
    # (9, 78),
    # (10, 93),
    # (11, 104),
    # (12, 128),
]

TIMEOUT = 60 * 60


def main() -> None:
    results = {}

    for n, LB in PUZZLE_NINE_INSTANCES:
        print(f"Validating LB={LB} for n={n}...")

        w_lb = w(n, 5, LB)
        width, height = w_lb + 2, w_lb + 2

        model_path = f"models/{n}x{LB}x{width}x{height}.txt"
        formula_path = f"formulas/{n}x{LB}x{width}x{height}.cnf"

        sat = run(
            name=f"Puzzle Nine {n}:{LB}",
            k=n,
            inside_min=LB,
            width=width,
            height=height,
            polyominoes=ALL_PENTOMINOES,
            break_global_symmetries=True,
            break_polyomino_symmetries=True,
            timeout=TIMEOUT,
            print_board=True,
            model_save_path=model_path,
            formula_save_path=formula_path,
        )

        if sat is None:
            results[n] = f"{LB} -> t"
            continue
        if not sat:
            results[n] = f"{LB} -> UNSAT"
            continue

        if sat:
            validate(
                k=n,
                inside_min=LB,
                width=width,
                height=height,
                polyominoes=ALL_PENTOMINOES,
                break_global_symmetries=True,
                break_polyomino_symmetries=True,
                model_save_path=model_path,
                formula_save_path=formula_path,
            )

        curr = LB + 1
        while True:
            w_lb = w(n, 5, curr)
            width, height = w_lb + 2, w_lb + 2

            model_path = f"models/{n}x{curr}x{width}x{height}.txt"
            formula_path = f"formulas/{n}x{curr}x{width}x{height}.cnf"

            print(f"Testing n={n}, LB={curr}...")

            sat = run(
                name=f"Puzzle Nine {n}:{curr}",
                k=n,
                inside_min=curr,
                width=width,
                height=height,
                polyominoes=ALL_PENTOMINOES,
                break_global_symmetries=True,
                break_polyomino_symmetries=True,
                timeout=TIMEOUT,
                print_board=False,
                model_save_path=model_path,
                formula_save_path=formula_path,
            )

            if sat is None:
                results[n] = f"{LB} -> t" if curr == LB + 1 else f"{LB} -> {curr - 1} (t)"
                break
            if not sat:
                results[n] = f"{LB} -> Upper Bound"
                break

            if sat:
                validate(
                    k=n,
                    inside_min=curr,
                    width=width,
                    height=height,
                    polyominoes=ALL_PENTOMINOES,
                    break_global_symmetries=True,
                    break_polyomino_symmetries=True,
                    model_save_path=model_path,
                    formula_save_path=formula_path,
                )

            curr += 1

    print("\n====== SUMMARY ======")
    for n, summary in results.items():
        print(f"{n}: {summary}")


if __name__ == "__main__":
    main()
