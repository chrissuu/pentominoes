import pytest

from src.Pentomino import ALL_PENTOMINOES
from src.PolyominoSolver import PolyominoSolver
from src.Tetromino import ALL_TETROMINOES


@pytest.mark.parametrize(
    "width,height,inside_tiles_minimum,k,polyominoes,is_satisfiable",
    [
        (10, 10, 9, None, ALL_TETROMINOES, True),
        (10, 10, 10, None, ALL_TETROMINOES, False),
        (8, 8, 6, 3, ALL_PENTOMINOES, True),
        (8, 8, 7, 3, ALL_PENTOMINOES, False),
        (10, 9, 13, 4, ALL_PENTOMINOES, True),
        (10, 9, 14, 4, ALL_PENTOMINOES, False),
        (9, 11, 24, 5, ALL_PENTOMINOES, True),
        (9, 11, 25, 5, ALL_PENTOMINOES, False),
        (12, 12, 34, 6, ALL_PENTOMINOES, True),
        (12, 12, 35, 6, ALL_PENTOMINOES, False),
    ],
)
def test_solver_cases(
    width: int,
    height: int,
    inside_tiles_minimum: int,
    k: int | None,
    polyominoes: list,
    is_satisfiable,
):
    solver = PolyominoSolver(
        k,
        inside_tiles_minimum,
        width,
        height,
        polyominoes,
        break_global_symmetries=True,
        break_polyomino_symmetries=True,
    )
    solvable = solver.solve()
    assert solvable == is_satisfiable


soln_20_20_128 = r"""OOOOOOOOOOOOOOOOOOOO
OOOOOOOOOFFFOOOOOOOO
OOOOOOFFFFIFFOOOOOOO
OOFFFFFIIIIIFFFOOOOO
OOFIIIIIIIIIIIFOOOOO
OOFIIIIIIIIIIIFFOOOO
OOFIIIIIIIIIIIIFOOOO
OOFIIIIIIIIIIIIFOOOO
OOFIIIIIIIIIIIIFOOOO
OOFIIIIIIIIIIIIFOOOO
OOFIIIIIIIIIIIFFFOOO
OOFIIIIIIIIIIIIFOOOO
OFFIIIIIIIIIIIIFOOOO
OOFFIIIIIIIIIIFFOOOO
OOOFIIIIIIIFFFFOOOOO
OOOFFFFFFFFFFOOOOOOO
OOOFOOOOFOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOO
OOOOOOOOOOOOOOOOOOOO""".splitlines()


def test_forced_soln_is_satisfiable():
    solver = PolyominoSolver(
        None,
        128,
        20,
        20,
        ALL_PENTOMINOES,
        break_global_symmetries=True,
        break_polyomino_symmetries=True,
    )

    # Force inside, outside, and fence cells to follow the known solution
    for x in range(solver.width):
        for y in range(solver.height):
            match soln_20_20_128[y][x]:
                case "F":
                    solver.cnf.append([solver.get_tf_var(x, y)])
                case "I":
                    solver.cnf.append([solver.get_ti_var(x, y)])
                case "O":
                    solver.cnf.append([solver.get_to_var(x, y)])

    satisfiable = solver.solve()
    assert satisfiable is True
