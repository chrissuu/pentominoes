"""
11/5
TODO:
1.) See if solution satisfies formula
    1.5) 
        ->instead of all pentominoes, consider subset
        ->grid size

2.) symmetry breaking
    2.5) 
        ->e.g., fix piece in top left quadrant
        ->e.g., constraint that some piece is within some quadrant

3.) fix piece + orientation + position of solution and see if solver can solve it

4.) send .cnf file

5.) duplicate tetrominoes and solve

6.) https://github.com/hgarrereyn/SBVA

7.) splitting clauses

8.) https://pentomino.classy.be/recorde.html

9.) gradually increasing difficulty could be useful for 
heuristically seeing how difficult the full original problem is

10.) for each cell: it's part of at most one piece
cardinality constraints with large bounds usually harder than with small bounds

11.) bug fix in 20,20,128 case
"""



"""
Constraints Edits: 11/4

1.) changed "adjacency" to also include diagonals. this enforces that the fence is properly enclosed 

2.) ensure that the outmost perimeter is an outside cell
"""

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

4.) Sum of inside cells should be greater than or equal to 129 (since we want to show 128 is maximal area)

5.) Handle all edge cases that arise....
"""
