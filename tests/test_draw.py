from pathlib import Path

from src.draw import board_to_img


def test_board_to_img():
    input_board_file = Path(__file__).parent / "input_board.txt"
    with open(input_board_file) as f:
        board = [list(line) for line in f.read().splitlines()]
    board_to_img(board)
