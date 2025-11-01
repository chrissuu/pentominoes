from typing import Sequence

from PIL import Image, ImageDraw


def hex2rgb(hex_color: str) -> tuple[int, ...]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


WHITE = hex2rgb("#ffffff")
BLACK = hex2rgb("#000000")
COLORS = {
    " ": WHITE,
    "F": hex2rgb("#ef4444"),  # Red
    "I": hex2rgb("#f97316"),  # Orange
    "L": hex2rgb("#eab308"),  # Yellow
    "N": hex2rgb("#84cc16"),  # Lime
    "P": hex2rgb("#34d399"),  # Emerald
    "T": hex2rgb("#06b6d4"),  # Cyan
    "U": hex2rgb("#3b82f6"),  # Blue
    "V": hex2rgb("#6366f1"),  # Indigo
    "W": hex2rgb("#a855f7"),  # Purple
    "S": hex2rgb("#d946ef"),  # Fuchsia
    "X": hex2rgb("#ec4899"),  # Pink
    "Y": hex2rgb("#64748b"),  # Slate
    "O": hex2rgb("#71717a"),  # Zinc
    "Z": hex2rgb("#78716c"),  # Stone
}


def board_to_img(
    board: Sequence[Sequence[str]],
    out_file: str = "board.png",
    cell_size: int = 64,
    grid_thickness: int = 1,
    outline_thickness: int = 3,
):
    rows, cols = len(board), len(board[0])

    # Create new image
    img = Image.new("RGBA", (cols * cell_size, rows * cell_size), WHITE)
    draw = ImageDraw.Draw(img)

    # Fill cells
    for r, row in enumerate(board):
        for c, char in enumerate(row):
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            draw.rectangle(
                [x0, y0, x1, y1], fill=COLORS[char], outline=BLACK, width=grid_thickness
            )

    # Draw thicker outlines between different-colored cells
    # Horizontal edges
    for r in range(rows + 1):
        for c in range(cols):
            if r == 0 or r == rows or board[r - 1][c] != board[r][c]:
                x0, y0 = c * cell_size, r * cell_size
                x1, y1 = x0 + cell_size, y0
                x0 -= outline_thickness // 2
                x1 += outline_thickness // 2
                draw.line([x0, y0, x1, y1], fill=BLACK, width=outline_thickness)
    # Vertical edges
    for r in range(rows):
        for c in range(cols + 1):
            if c == 0 or c == cols or board[r][c - 1] != board[r][c]:
                x0, y0 = c * cell_size, r * cell_size
                x1, y1 = x0, y0 + cell_size
                y0 -= outline_thickness // 2
                y1 += outline_thickness // 2
                draw.line([x0, y0, x1, y1], fill=BLACK, width=outline_thickness)

    # Save and show image
    img.save(out_file)
    img.show()
    print(f"Saved grid image as '{out_file}'")
