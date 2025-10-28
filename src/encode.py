"""
[IDEA] The high level idea is as follows:

1.) Fix some arbitrary rectangular enclosure where pentominoes can be placed.
    For now, suppose this is an m x n rectangle.

2.) Have three boolean variables denoting whether cell (x,y) is a fence, inside cell, or outside cell.
    
    C_f(x,y) := if cell (x,y) is a fence cell
    C_i(x,y) := if cell (x,y) is an inside cell
    C_o(x,y) := if cell (x,y) is an outside cell

    *---------------------*
    |                     |
    |         ***         |
    |        **I*         |
    |        *II*         |
    |        ****         |
    |                     |
    *---------------------*

    Here, blank cells are "outside cells", "I" are "inside cells" and * are pentominoes or fence cells.

3.) Fix some corner of a pentomino as well as the default configuration at the beginning of the procedure. 
    We will use this corner to figure out where the rest of the pentomino is relative to the corner cell.

    This also means we need a boolean variable(s):

        P(x,y,r,i) := if cell (x,y) has pentomino i (fixed to some corner), with rotation r, 0<=r<=3

    Which carries the information of which pentomino, how many rotations from default configuration,
    placed at what cell.

    We can use this to propagate information and figure out which other cells are occupied by 
    the pentomino. 
    
    For instance, if pentomino 1 := 

        * *
        ***

    And we fixed a corner at the bottom left piece, then 
    
        P(0, 0, 0, 1) => P^(0, 1, 1) and P^(1, 0, 1) and P^(2, 0, 1) and P^(2, 1, 1)

Constraints:
1.) To ensure that you end up with one connected component, define a single "sink" boolean variable.
    We will enforce that for all cells which are inside cells, the "sink" boolean variable is
    reachable from all the inside cells.

    This implies that if two (or more) connected components exist, then one of the connected components
    will not have a sink variable, and hence the inside cells of that connected component are not sink-reachable.

2.) Ensure that each pentomino is placed exactly once

3.) Ensure that if you are an outside cell, then your adjacent cells are either an outside cell or a fence cell. 

4.) Handle all edge cases that arise....
"""