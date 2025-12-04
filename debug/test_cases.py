# test_polyomino_solver.py
"""
PolyominoSolver Test Suite
Tests the maximal tetromino area enclosing problem
on different board sizes and tile configurations.
"""
from pentominoes import PolyominoSolver, ALL_TETROMINOES, ALL_PENTOMINOES

def run_test_case(name, width, height, inside_min, polyominoes, expected):
    print(f"\n=== TEST CASE: {name} ===")
    solver = PolyominoSolver(width, height, inside_min, polyominoes)
    success = solver.solve()
    if success == expected:
        if success:
            solver.print_board()
        print(f"Passed test case {name}")
    else:
        print(f"Failed test case {name}. Got {success} when expecting {expected}.")
        assert(False)
    print("=" * 40 + "\n")

T1 = lambda _ : run_test_case(
    name="All Tetrominoes sat",
    width=10,
    height=10,
    inside_min=9,
    polyominoes=ALL_TETROMINOES,
    expected=True
)

T2 = lambda _ : run_test_case(
    name="All Tetrominoes unsat",
    width=10,
    height=10,
    inside_min=10,
    polyominoes=ALL_TETROMINOES,
    expected=False
)

T3 = lambda _ : run_test_case(
    name="Corners pieces",
    width=10,
    height=10,
    inside_min=1,
    polyominoes=[L4(), N4()],
    expected=False
)

T4 = lambda _ : run_test_case(
    name="Duplicates",
    width=10,
    height=10,
    inside_min=1,
    polyominoes=[L4(), L4()],
    expected=True
)

T6 = lambda _ : run_test_case(
    name="Double Tetrominoes 10",
    width=18,
    height=18,
    inside_min=10,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

T7 = lambda _ : run_test_case(
    name="Double Tetrominoes 15",
    width=18,
    height=18,
    inside_min=15,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

# 36.45 seconds
T8 = lambda _ : run_test_case( 
    name="Double Tetrominoes 40",
    width=18,
    height=18,
    inside_min=40,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

# 63.86 seconds
# 100961, 280150
T9 = lambda _ : run_test_case(
    name="Double Tetrominoes 41",
    width=18,
    height=18,
    inside_min=41,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

# 186.39 seconds
# 101202, 280634
T10 = lambda _ : run_test_case(
    name="Double Tetrominoes 42",
    width=18,
    height=18,
    inside_min=42,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

# 70.05
# 101441, 281114
T11 = lambda _ : run_test_case(
    name="Double Tetrominoes 43",
    width=18,
    height=18,
    inside_min=43,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

T12 = lambda _ : run_test_case(
    name="Double Tetrominoes 80",
    width=18,
    height=18,
    inside_min=80,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

T12 = lambda _ : run_test_case(
    name="Double Tetrominoes 80",
    width=18,
    height=18,
    inside_min=80,
    polyominoes=ALL_TETROMINOES * 2,
    expected=True
)

T13 = lambda _ : run_test_case(
    name="All Pentominoes 128",
    width=20,
    height=20,
    inside_min=120,
    polyominoes=ALL_PENTOMINOES,
    expected=True
)

T14 = lambda _ : run_test_case(
    name="All Pentominoes 128",
    width=20,
    height=20,
    inside_min=128,
    polyominoes=ALL_PENTOMINOES,
    expected=True
)
