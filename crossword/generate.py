from collections import deque
import copy
import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox(
                            (0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable, words in self.domains.items():
            self.domains[variable] = {
                word for word in words if len(word) == variable.length}

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revision = False

        # First find the position of variable x and y that overlaps each other
        if overlap := self.crossword.overlaps.get((x, y)):
            # Make a set with words to remove after looping. Check if the overlapping character in word a
            # has no matches in possible words for y
            remove_words = {word for word in self.domains.get(x) if all(
                word[overlap[0]] != y_word[overlap[1]] for y_word in self.domains.get(y))}

            # Only if there are no false values, remove the word
            if remove_words:
                revision = True

                # Update x's domain
                self.domains[x] -= remove_words

        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        # Check if arcs == None
        if arcs == None:
            arcs = deque(
                (x, y) for x in self.crossword.variables for y in self.crossword.neighbors(x))

        # In case arcs is not already deque
        elif not isinstance(arcs, deque):
            arcs = deque(arcs)

        # Loop as long as the queu exists
        while arcs:
            (x, y) = arcs.pop()
            if self.revise(x, y):
                if not self.domains.get(x):
                    return False
                arcs.extend((n, x) for n in self.crossword.neighbors(
                    x) if n != y and (n, x) not in arcs)

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return all(var in assignment for var in self.crossword.variables)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # Check if all words are unique. Set deletes duplicates, so if the length changes,
        # then there were duplicates
        words = assignment.values()
        if len(words) != len(set(words)):
            return False

        # Make sure that any neighbour in assignment, that has overlaps with the variable,
        # has the same character in the overlapping position
        for var, word in assignment.items():
            if len(word) != var.length:
                return False

            for n in self.crossword.neighbors(var):
                if n not in assignment or not (overlap := self.crossword.overlaps.get((var, n))):
                    continue
                if word[overlap[0]] != assignment[n][overlap[1]]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        return sorted(
            # Return a list (ref requirement) of all word in var's domain
            list(self.domains.get(var)),
            key=lambda word: sum(
                # Check if the characters match in the overlapping field
                word[overlap[0]] != neigbour_word[overlap[1]]
                for neighbour in self.crossword.neighbors(var)
                # If there are any overlaps with var and neighbour
                if (overlap := self.crossword.overlaps.get((var, neighbour)))
                # Only count for this neighbour if it does not already have an assigned value
                and neighbour not in assignment
                for neigbour_word in self.domains.get(neighbour)
            )
        )

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Key is a tuple to use min-functions secondary selection, if necessary
        # Minimum remaining values: first value in the tuple. Number of neighbors: second value
        return min((var for var in self.crossword.variables if var not in assignment),
                   key=lambda var: (len(self.domains.get(var)), -len(self.crossword.neighbors(var))))

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)

        if var is not None:
            for value in self.order_domain_values(var, assignment):
                new_assignment = assignment.copy()
                new_assignment[var] = value

                # If it is consistent, then loop through backtrack
                if self.consistent(new_assignment):

                    # Copy the current domains, as ac3/revise will make changes here. These needs to be undone
                    # if the new_assignment fails
                    domains_copy = copy.deepcopy(self.domains)
 
                    # Call ac3 again with neighbors to go through arc consistency after making an assignment
                    if self.ac3([(neighbor, var) for neighbor in self.crossword.neighbors(var)]) == True:
                        result = self.backtrack(new_assignment)

                        # If the backtrackin goes all the way to the end and
                        # finds a complete assignment, this assignment should be returned
                        if result is not None:
                            return result
                    
                    # Restore domains if backtracking
                    self.domains = domains_copy  

        # If either of the recurrent loops returned None, the final result will be None
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
