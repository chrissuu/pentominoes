from typing import Sequence
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

def hex2rgb(hex_color: str) -> tuple[int, ...]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

WHITE = hex2rgb("#ffffff")
BLACK = hex2rgb("#000000")
GRAY = hex2rgb("#6b7280")  # Medium gray for label
BORDER_CHAR = '*'
EMPTY_CHAR = '-'
INSIDE_CHAR = '+'

COLORS = {
    "-": WHITE,
    "+": hex2rgb("#fef3c7"),  # Light cream for inside
    "F": hex2rgb("#ff6b6b"),  # Bright Red
    "I": hex2rgb("#ffa94d"),  # Bright Orange
    "L": hex2rgb("#ffd43b"),  # Bright Yellow
    "N": hex2rgb("#a0d911"),  # Bright Lime
    "P": hex2rgb("#51cf66"),  # Bright Green
    "T": hex2rgb("#22d3ee"),  # Bright Cyan
    "U": hex2rgb("#4dabf7"),  # Bright Blue
    "V": hex2rgb("#748ffc"),  # Bright Indigo
    "W": hex2rgb("#b197fc"),  # Bright Purple
    "S": hex2rgb("#da77f2"),  # Bright Magenta
    "X": hex2rgb("#f783ac"),  # Bright Pink
    "Y": hex2rgb("#8c92ac"),  # Soft Slate
    "O": hex2rgb("#96989d"),  # Soft Gray
    "Z": hex2rgb("#a89f94"),  # Soft Brown
}

def parse_board(board_str: str) -> list[list[str]]:
    """Parse board string into 2D array, trimming border."""
    lines = [line.strip() for line in board_str.strip().split('\n') if line.strip()]
    
    # Remove outer border (*)
    trimmed = []
    for line in lines[1:-1]:  # Skip first and last row
        trimmed.append(list(line[1:-1]))  # Skip first and last column
    
    # Find smallest enclosing rectangle
    min_r, max_r = len(trimmed), -1
    min_c, max_c = len(trimmed[0]) if trimmed else 0, -1
    
    for r in range(len(trimmed)):
        for c in range(len(trimmed[0])):
            if trimmed[r][c] != EMPTY_CHAR:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    # Add one border of white space around the enclosing rectangle
    min_r = max(0, min_r - 1)
    max_r = min(len(trimmed) - 1, max_r + 1)
    min_c = max(0, min_c - 1)
    max_c = min(len(trimmed[0]) - 1, max_c + 1)
    
    # Extract the trimmed board
    result = []
    for r in range(min_r, max_r + 1):
        result.append(trimmed[r][min_c:max_c + 1])
    
    return result

def get_region_info(board: list[list[str]]) -> dict:
    """Get area of each colored region using flood fill."""
    rows, cols = len(board), len(board[0])
    visited = [[False] * cols for _ in range(rows)]
    regions = defaultdict(int)
    
    def flood_fill(r, c, char):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0
        if visited[r][c] or board[r][c] != char:
            return 0
        
        visited[r][c] = True
        count = 1
        count += flood_fill(r + 1, c, char)
        count += flood_fill(r - 1, c, char)
        count += flood_fill(r, c + 1, char)
        count += flood_fill(r, c - 1, char)
        return count
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and board[r][c] not in [EMPTY_CHAR, INSIDE_CHAR]:
                char = board[r][c]
                area = flood_fill(r, c, char)
                if char not in regions:
                    regions[char] = area
    
    return regions

def get_inside_area(board: list[list[str]]) -> int:
    """Count cells marked with '+'."""
    count = 0
    for row in board:
        count += row.count(INSIDE_CHAR)
    return count

def darken_color(color: tuple, factor: float = 0.7) -> tuple:
    """Darken a color by a factor."""
    return tuple(int(c * factor) for c in color)

def board_to_img(
    board_str: str,
    k: int,
    out_file: str = "board.png",
    cell_size: int = 64,
    grid_thickness: int = 1,
    outline_thickness: int = 3,
    label_height: int = 80,
):
    board = parse_board(board_str)
    rows, cols = len(board), len(board[0])
    
    # Get region info
    regions = get_region_info(board)
    inside_area = get_inside_area(board)
    
    # Create new image with space for label (no background for label area)
    img = Image.new("RGBA", (cols * cell_size, rows * cell_size + label_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw label at top
    try:
        label_font = ImageFont.truetype("ariali.ttf", 60)  # Italic
        area_font = ImageFont.truetype("arialbd.ttf", 72)  # Increased from 48 to 72
    except:
        try:
            label_font = ImageFont.truetype("Arial Italic.ttf", 60)
            area_font = ImageFont.truetype("Arial Bold.ttf", 72)  # Increased from 48 to 72
        except:
            label_font = ImageFont.load_default()
            area_font = ImageFont.load_default()
    
    label_text = f"k = {k}"
    bbox = draw.textbbox((0, 0), label_text, font=label_font)
    text_width = bbox[2] - bbox[0]
    text_x = (cols * cell_size - text_width) // 2
    draw.text((text_x, 15), label_text, fill=GRAY, font=label_font)
    
    # Offset for board (below label)
    y_offset = label_height
    
    # Fill cells with gradient shading
    for r, row in enumerate(board):
        for c, char in enumerate(row):
            x0, y0 = c * cell_size, r * cell_size + y_offset
            x1, y1 = x0 + cell_size, y0 + cell_size
            
            base_color = COLORS.get(char, WHITE)
            
            # Create gradient effect for colored pieces
            if char not in [EMPTY_CHAR, INSIDE_CHAR]:
                # Draw gradient: lighter at top, darker at bottom
                for i in range(cell_size):
                    factor = 1.0 - (i / cell_size) * 0.3
                    gradient_color = tuple(int(c * factor) for c in base_color)
                    draw.line([(x0, y0 + i), (x1, y0 + i)], fill=gradient_color, width=1)
            elif char == INSIDE_CHAR:
                # Checkered pattern for inside area
                pattern_size = 8
                for py in range(cell_size):
                    for px in range(cell_size):
                        if ((px // pattern_size) + (py // pattern_size)) % 2 == 0:
                            draw.point((x0 + px, y0 + py), fill=base_color)
                        else:
                            draw.point((x0 + px, y0 + py), fill=darken_color(base_color, 0.95))
            else:
                draw.rectangle([x0, y0, x1, y1], fill=base_color)
            
            # Draw grid
            draw.rectangle([x0, y0, x1, y1], outline=BLACK, width=grid_thickness)
    
    # Draw area number for inside region
    if inside_area > 0:
        # Find center of inside region
        inside_cells = []
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == INSIDE_CHAR:
                    inside_cells.append((r, c))
        
        if inside_cells:
            avg_r = sum(r for r, c in inside_cells) / len(inside_cells)
            avg_c = sum(c for r, c in inside_cells) / len(inside_cells)
            center_x = int(avg_c * cell_size + cell_size // 2)
            center_y = int(avg_r * cell_size + cell_size // 2 + y_offset)
            
            area_text = str(inside_area)
            bbox = draw.textbbox((0, 0), area_text, font=area_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = center_x - text_width // 2
            text_y = center_y - text_height // 2
            
            # Draw white border around text
            border_width = 3
            for dx in range(-border_width, border_width + 1):
                for dy in range(-border_width, border_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), 
                                 area_text, fill=WHITE, font=area_font)
            
            # Draw main text in black
            draw.text((text_x, text_y), 
                     area_text, fill=BLACK, font=area_font)
    
    # Draw thicker outlines between different regions
    for r in range(rows + 1):
        for c in range(cols):
            if r == 0 or r == rows or board[r - 1][c] != board[r][c]:
                x0, y0 = c * cell_size, r * cell_size + y_offset
                x1, y1 = x0 + cell_size, y0
                draw.line([x0, y0, x1, y1], fill=BLACK, width=outline_thickness)
    
    for r in range(rows):
        for c in range(cols + 1):
            if c == 0 or c == cols or board[r][c - 1] != board[r][c]:
                x0, y0 = c * cell_size, r * cell_size + y_offset
                x1, y1 = x0, y0 + cell_size
                draw.line([x0, y0, x1, y1], fill=BLACK, width=outline_thickness)
    
    img.save(out_file)
    print(f"Saved grid image as '{out_file}'")
    print(f"Inside area: {inside_area}")
    print(f"Unique pieces (k): {k}")
    return img


def process_log_file(log_file_path: str, output_image_path: str):
    """
    Parse a log file and generate a board image from it.
    
    Args:
        log_file_path: Path to the log file containing the puzzle results
        output_image_path: Path where the output image should be saved
    
    Returns:
        True if successful, False if UNSATISFIABLE
    """
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Check if result is UNSATISFIABLE
    if "UNSATISFIABLE" in content:
        print(f"Log file shows UNSATISFIABLE result - no board to generate")
        return False
    
    # Extract k value from parameters
    k = None
    for line in content.split('\n'):
        if '- k:' in line:
            k = int(line.split(':')[1].strip())
            break
    
    if k is None:
        raise ValueError("Could not find 'k' parameter in log file")
    
    # Extract board (between the asterisk borders)
    lines = content.split('\n')
    board_lines = []
    in_board = False
    
    for line in lines:
        if line.startswith('**********'):
            if not in_board:
                in_board = True
                board_lines.append(line)
            else:
                board_lines.append(line)
                break
        elif in_board:
            board_lines.append(line)
    
    if not board_lines:
        raise ValueError("Could not find board in log file (result may be UNSATISFIABLE)")
    
    board_str = '\n'.join(board_lines)
    
    # Generate the image
    board_to_img(board_str, k, out_file=output_image_path)
    print(f"Successfully processed log file and saved image to '{output_image_path}'")
    return True

if __name__ == "__main__":
    import os

    for file_dir in sorted(os.listdir('logs')):
        metadata = file_dir.split('_')
        k = int(metadata[-2])
        inside_area = int(metadata[-1].split('.')[0])
        process_log_file(f'logs/{file_dir}', f'boards/{k}x{inside_area}.png')