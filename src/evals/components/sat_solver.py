import os

from symai import Symbol, Expression
from symai.components import FileReader, Execute

from z3 import Solver, sat


cur_file_dir = os.path.dirname(os.path.abspath(__file__))


LOGIC_TEMPLATE = """
# imports the available functions from the z3 library
from z3 import Solver, Function, IntSort, EnumSort, Int, And, Or, Xor, Const

# Define the problem statement as a function that takes a solver as input and returns a query constant as output
def problem_statement(S: Solver) -> Const:
    # Example for using the solver:
    # Porp, (A, B, C) = EnumSort('Prop', ('A', 'B', 'C')) # Define an enumerated sort
    # p = Function('prop_func', IntSort(), Prop)          # Define an uninterpreted function that takes an integer as input and returns a Prop as output
    # S.add(B == p(2))                                    # Assert a new fact
    # S.add(And(p(1) == A, p(2) == B, p(3) == C))         # Assert a new fact
    # ...                                                 # Define more facts
    # query = Const("query", Prop)                        # Create a new constant
    #
    # TODO: Define the logic expressions here using the S variable as the solver.
    query = None
    # insert your code here
    return query

# assign result to global output variable to make accessible for caller
_value_obj_ = problem_statement
"""


class SATSolver(Expression):
    def forward(self, code, presets):
        solution  = presets()
        # Create the execution template
        runner    = Execute(enclosure=True)
        # Execute the code
        statement = runner(code)
        # Create a new solver instance
        S         = Solver()
        # Create a new query
        query     = statement['locals']['_output_'](S)
        # Check if the query can be solved
        r         = S.check()
        # Print the solution
        if r == sat:
            # Get the model
            m = S.model()
            # Return the solution
            return str(m[query]) == str(solution)
        else:
            print("Cannot solve the puzzle. Returned: " + str(r))
            return False


if __name__ == '__main__':
    solver = SATSolver()
    reader = FileReader()
    ori    = reader(os.path.join(cur_file_dir, '../snippets/einstein_puzzle_logic_solution.txt'))
    result = solver(ori, lambda: 'German')
    print(result)
