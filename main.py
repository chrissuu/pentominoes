from pentominoes import ALL_PENTOMINOES, run, validate, w

seconds = int

"""
NEW LOWER BOUNDS
"""
# PUZZLE_NINE_INSTANCES = [
#     (3, 6),
#     (4, 12),
#     (5, 21),
#     (6, 34),
#     (7, 47),
#     (8, 62),
#     (9, 78),
#     (10, 93),
#     (11, 104),
#     (12, 128),
# ]

"""
ORIGINAL LOWER BOUNDS
"""
PUZZLE_NINE_INSTANCES = [
    (3, 6),
    (4, 12),
    (5, 21),
    (6, 32),
    (7, 43),
    (8, 61),
    (9, 70),
    (10, 84),
    (11, 102),
    (12, 128),
]

TIMEOUT = 60 * 60

def main() -> None:
    results = {}

    for n, LB in PUZZLE_NINE_INSTANCES:
        print(f"Validating LB={LB} for n={n}...")
        record = {
            "initial": LB,
            "best": None,
            "ub": None,
            "timeout_at": None,
            "status": "unknown",
        }

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
            record["status"] = "timeout_initial"
            results[n] = record
            continue
        if not sat:
            record["status"] = "initial_unsat"
            record["ub"] = LB
            results[n] = record
            continue

        record["best"] = LB
        record["status"] = "searching"

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
                print_board=True,
                model_save_path=model_path,
                formula_save_path=formula_path,
            )

            if sat is None:
                record["status"] = "timeout"
                record["timeout_at"] = curr
                results[n] = record
                break
            if not sat:
                record["status"] = "ub_proven"
                record["ub"] = curr
                results[n] = record
                break

            record["best"] = curr

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
    if results:
        def format_lb_progress(record):
            initial = record["initial"]
            best = record["best"]
            status = record["status"]
            if best is None:
                if status == "initial_unsat":
                    return f"{initial} (UNSAT)"
                return f"{initial} (unproved)"
            if best == initial:
                return f"{initial}"
            return f"{initial} -> {best}"

        def format_ub_status(record):
            if record["ub"] is not None:
                if record["status"] == "initial_unsat":
                    return f"proved at {record['ub']} (initial UNSAT)"
                return f"proved at {record['ub']}"
            status = record["status"]
            if status == "timeout":
                return f"pending (timeout at {record['timeout_at']})"
            if status == "timeout_initial":
                return "pending (timeout before SAT)"
            return "pending"

        rows = []
        for n in sorted(results):
            record = results[n]
            rows.append(
                (
                    str(n),
                    format_lb_progress(record),
                    format_ub_status(record),
                )
            )

        n_width = max(len(row[0]) for row in rows)
        lb_width = max(len("LB progression"), max(len(row[1]) for row in rows))
        ub_width = max(len("Upper bound"), max(len(row[2]) for row in rows))

        header = (
            f"{'n'.rjust(n_width)} | "
            f"{'LB progression'.ljust(lb_width)} | "
            f"{'Upper bound'.ljust(ub_width)}"
        )
        divider = (
            f"{'-' * n_width}-+-{'-' * lb_width}-+-{'-' * ub_width}"
        )
        print(header)
        print(divider)
        for row in rows:
            print(
                f"{row[0].rjust(n_width)} | "
                f"{row[1].ljust(lb_width)} | "
                f"{row[2].ljust(ub_width)}"
            )
    else:
        print("No results recorded.")


if __name__ == "__main__":
    main()
