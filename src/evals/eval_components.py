# good reference: https://ericpony.github.io/z3py-tutorial/guide-examples.htm

from pathlib import Path

from symai import Expression, Symbol
from symai.components import Execute, FileReader, Function
from symai.extended import Conversation
from symai.post_processors import CodeExtractPostProcessor, StripPostProcessor
from symai.utils import toggle_test

from src.utils import MOCK_RETURN, RANDOM_SEQUENCE, normalize

ACTIVE = True


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_solve_puzzle():
    scoring  = []
    reader   = FileReader()
    dir_path = Path(__file__).parent.absolute() / "snippets"

    problem        = reader((dir_path / "einstein_puzzle.txt").as_posix())
    solution       = reader((dir_path / "einstein_puzzle_human_solution.txt").as_posix())
    human_solution = reader((dir_path / "jays_brother_human_solution.txt").as_posix())
    trajectories   = reader((dir_path / "jays_brother_trajectories.txt").as_posix())

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

    # Solve the puzzle
    cv     = Conversation(auto_print=True)
    res    = cv(template)
    code   = pp(res.value, None, tag="python")

    # Execute the code
    try:
        exec(
            code,
            globals()
        )

        answer = globals().get("answer")

        if answer is not None and "John" in answer:
            # Attaboy!
            scoring.append(1.0)
        else:
            # No cigar
            scoring.append(0.0)
    except Exception as e:
        scoring.append(0.0)

    # How good?
    random     = Symbol(RANDOM_SEQUENCE)
    rand_score = human_solution.similarity(random, metric='cosine')
    base_score = human_solution.similarity(trajectories.split("\n\n\n")).mean()
    score      = human_solution.similarity(code, normalize=normalize(base_score, rand_score))
    scoring.append(score)

    return True, {'scores': scoring}

if __name__ == "__main__":
    answer = test_solve_puzzle()
    print(answer)


