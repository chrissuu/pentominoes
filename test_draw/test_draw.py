from pathlib import Path

import pytest

from draw import board_to_img


@pytest.mark.parametrize(
    "input_board_file",
    [
        Path(__file__).parent / "input_board_tetrominoes.txt",
        Path(__file__).parent / "input_board_pentominoes.txt",
    ],
)
def test_board_to_img(input_board_file):
    with open(input_board_file) as f:
        board = [list(line) for line in f.read().splitlines()]
    board_to_img(board)
