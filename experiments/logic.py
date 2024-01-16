# good reference: https://ericpony.github.io/z3py-tutorial/guide-examples.htm

from pathlib import Path

from symai import Expression, Symbol
from symai.components import Execute, FileReader, Function
from symai.extended import Conversation
from symai.post_processors import CodeExtractPostProcessor, StripPostProcessor
from z3 import Solver, sat


def test_solve_puzzle():
    reader   = FileReader()
    problem  = reader("experiments/einstein_puzzle.txt")
    solution = reader("experiments/einstein_puzzle_solution.txt")

    query = "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother?"
    task  = """
    Using the provided solution to the Einstein puzzle as a reference, apply the Z3 solver to solve the puzzle.
    Return executable code following the markdown convention that upon running is solving the puzzle and assigning the result to an `answer` variable.
    """

    template = f"""
    [Problem]
    {problem}

    [Solution]
    {solution}

    [Task]
    {task}

    [Puzzle]
    {query}
    """

    # Nicely deal with the markdown
    pp = CodeExtractPostProcessor()

    # Execute the code
    cv = Conversation(auto_print=True)
    res = cv(template)

    # Extract the code
    try:
        exec(
            pp(res.value, None, tag="python"),
            globals()
        )
        answer = Symbol(globals()['answer'])
        return "John" in answer.value

    except Exception as e:
        return False


if __name__ == "__main__":
    answer = test_solve_puzzle()
