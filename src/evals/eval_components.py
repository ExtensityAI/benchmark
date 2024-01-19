import os
from pathlib import Path

import sympy as sym
from symai import Conversation, Expression, Function, Symbol
from symai.components import FileReader
from symai.extended import Conversation
from symai.post_processors import CodeExtractPostProcessor, StripPostProcessor
from symai.processor import ProcessorPipeline
from symai.utils import toggle_test

from src.evals.components import Factorization
from src.evals.components.sat_solver import LOGIC_TEMPLATE, SATSolver
from src.utils import MOCK_RETURN, RANDOM_SEQUENCE, normalize

from z3 import Solver, sat

ACTIVE = False

cur_file_dir = os.path.dirname(os.path.abspath(__file__))


HOL_FACTORIZATION_CONTEXT = """[Context]
Factorize a given linguistic expression by creating logical components that transform the statement into compositional SAT statement.

[Boolean Operations]
AND       ... logical and
OR        ... logical or
XOR       ... logical xor
NOT(x)    ... logical not
HAS(x, y) ... binary relation where x has y (for symmetric relations, use has(x, y) AND has(y, x))
IS(x, y)  ... binary relation where x is an instance of y (for symmetric relations, use is(x, y) AND is(y, x))

[Primitive Types]
TRUE      ... boolean true
FALSE     ... boolean false
STRING    ... a string literal

Any other words are considered as variables. Variables with a parenthesis '(...)' are considered as functions and will be resolved by the solver engine i.e. Furry(X) will be resolved by the solver engine as a function Furry(X) and may return TRUE or FALSE. Variables are written in lower case, functions are written in Pascal case and primitive types are written in upper case. Function can have multiple arguments but must be separated by a comma ',' and enclosed in parenthesis '(...)', and return a boolean value. Arguments for functions or operations can only be variables, not primitive types. Primitive types can be used as terminal symbols to describe the meaning of a variable.
The '<-' operator is used to assign the result of the right hand side to the left hand side. The right hand side can be a logical expression or a function call or a primitive such as TRUE, FALSE, literal string. The left hand side can be a variable or a function name. '//' can be used to introduce comments.

If a new variable or function is introduced, provide either a definition using the '<-' operator or for terminal symbols provide a description using the ':' operator, but not both! The quotes '"' are used to define terminal symbol description. Functions must be resolvable as boolean logic either by returning 'TRUE' or 'FALSE' or by posing a question that is resolvable by as a true or false statement. All variables must be defined either as a function or as a terminal symbol description. Terminate a statement with a ';' operator. All expressions are enclosed in ```expression ... ``` blocks.

[Example]
```expression
furry(x) <- HAS(x, fur) OR HAS(x, hair)
fur: "a soft, hairy covering on the skin of an animal, such as a fox or beaver, consisting of a network of nonliving keratin cells and embedded in a layer of dermis";
hair: "any of the fine threadlike strands growing from the skin of humans, mammals, and some other animals";
```

Here is a more complete example:

[Example]
Marvins has four pawns and likes to meow when I pet its fur. Is Marvins a cat?

[Solution]
```expression
Cat(x) <- Furry(x) AND Meows(x) AND HAS(x, paws);
paws: "a clawed foot of an animal, especially a quadruped, that has claws or nails";
Meows(x): "does x produce a characteristic crying sound?";
Furry(x): HAS(x, fur) OR HAS(x, hair);
fur: "a soft, hairy covering on the skin of an animal, such as a fox or beaver, consisting of a network of nonliving keratin cells and embedded in a layer of dermis";
hair: "any of the fine threadlike strands growing from the skin of humans, mammals, and some other animals";

// usage
Cat("Marvins");
```

Here X = Marvins, which evaluates the query Cat(Marvins) to a true statement.
The evaluation of the above statements will be interpreted as a sequence of instructions that can be resolved with an pre-defined set of functions or solver engine.

[Task]
Factorize the following expression:
"""


class HOLFactorization(Function):
    @property
    def static_context(self):
        return HOL_FACTORIZATION_CONTEXT


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_factorize_formula():
    a, b, c, d, x, y = sym.symbols('a, b, c, d, x, y')
    expr        = a * x + b * x - c * x - a * y - b * y + c * y + d
    stmt        = Symbol("Can you simplify me the following expression: a*x + b*x - c*x - a*y - b*y + c*y + d")
    res         = stmt.extract('formula')
    #res goes to sympy
    symbols_    = stmt.extract('all unique symbols as a list')
    fact        = sym.collect(expr, d, func=sym.factor)
    # model based factorization
    func        = Factorization('Factorize d from the expression such that your final start with: `d + (...`:')
    res         = func(expr)
    ref         = Symbol(fact)
    random      = Symbol(RANDOM_SEQUENCE)
    rand_score  = ref.similarity(random)
    base_score  = ref.similarity([Symbol("The factorized result is: d+(a+b-c)*(x-y)"),
                                  Symbol("We obtain: d + ( x - y ) * ( a + b - c )"),
                                  Symbol("(a + b - c) * (x - y) + d")]).mean()
    # validate
    score       = ref.similarity(res, normalize=normalize(base_score, rand_score))
    return True, {'scores': [score]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_dsl_writing_capability():
    # test only the capability to follow instructions from a custom DSL (syntax) + semantic structure
    solution1 = """
// Query
IsBrotherOf(jay, john, bob) <- BrotherOf(jay, john) AND FatherOf(bob, jay) AND FatherOf(bob, john);

// Facts
BrotherOf(x, y) <- HAS(x, brother) AND HAS(y, brother) AND Sibling(x, y);
FatherOf(x, y) <- HAS(x, son) AND ParentOf(x, y);
ParentOf(x, y) <- IS(x, parent) AND IS(y, child);
Sibling(x, y) <- IS(x, father) AND IS(y, father) OR IS(x, mother) AND IS(y, mother);

// Primitive Types
son: "a male child in relation to his parents";
father: "a male parent";
mother: "a female parent";
brother: "a male sibling";
parent: "a person's father or mother";
child: "a young human being below the legal age of majority associated to this person as a parent";
"""

    solution2 = """
IsBrotherOf(x, y, z) <- BrotherOf(x, y) AND FatherOf(z, x) AND FatherOf(z, y);
BrotherOf(x, y) <- Sibling(x, y) AND IS(x, brother) AND IS(y, brother);
FatherOf(x, y) <- ParentOf(x, y) AND IS(y, son);
Sibling(x, y) <- CommonParent(x, y);
CommonParent(x, y) <- (IS(x, father) AND IS(y, father)) OR (IS(x, mother) AND IS(y, mother));
IS(x, brother) <- TRUE; // Implied by the use of 'x, brother' and 'y, brother'
IS(y, brother) <- TRUE;
IS(y, son) <- TRUE;
IS(x, father): "is x acknowledged as a father of someone?";
IS(x, mother): "is x acknowledged as a mother of someone?";
IS(x, parent): "is x acknowledged as a parent of someone?";
IS(x, child): "is x acknowledged as a child of someone?";
ParentOf(x, y) <- IS(x, parent) AND IS(y, child);
"""
    val  = "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother."
    scoring     = []
    expr        = HOLFactorization(val, post_processors=[StripPostProcessor(), CodeExtractPostProcessor()])
    res         = expr(val)
    sol1        = Symbol(solution1)
    sol2        = Symbol(solution2)
    random      = Symbol(RANDOM_SEQUENCE+val) # remove the chance of simply rephrasing the question
    rand_score  = random.similarity([sol1, sol2]).mean()
    base_score  = sol1.similarity(sol2)
    score       = sol1.similarity(res, normalize=normalize(base_score, rand_score))
    scoring.append(score)
    # check for syntax violations
    if '("' in str(res) or '")' in str(res) or '",' in str(res) or '":' in str(res) or '=' in str(res):
        scoring.append(0.0)
        return False, {'scores': scoring}
    else:
        scoring.append(1.0)
        return True, {'scores': scoring}


def test_solve_puzzle():
    problem   = """
The Englishman lives in the red house.
The Swede keeps dogs.
The Dane drinks tea.
The green house is just to the left of the white one.
The owner of the green house drinks coffee.
The Pall Mall smoker keeps birds.
The owner of the yellow house smokes Dunhills.
The man in the center house drinks milk.
The Norwegian lives in the first house.
The Blend smoker has a neighbor who keeps cats.
The man who smokes Blue Masters drinks bier.
The man who keeps horses lives next to the Dunhill smoker.
The German smokes Prince.
The Norwegian lives next to the blue house.
The Blend smoker has a neighbor who drinks water.
The question to be answered is: Who keeps fish?
"""
    task      = """[Task]
Implement the `problem_statement` function that takes in the z3 package `S` solver as input and returns a query constant as output.
All required imports are already provided. The code of the `problem_statement` function should be written between a
```python
...
```
code block.
The `problem_statement` function must be self-contained, fully functional and pass all tests.
No other functions or explanations are required.
The implementation must be a SAT solvable solution and follow the user problem requirements:

[Problem Statement]
%s
""" % problem
    conv       = Conversation(init=LOGIC_TEMPLATE, auto_print=False)
    res        = conv(task)
    scoring    = []
    processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])
    code       = Symbol(processors(str(res), None))
    reader     = FileReader()
    solution   = reader(os.path.join(cur_file_dir, 'snippets/einstein_puzzle_logic_solution.txt'))
    sim        = solution.similarity(res)
    scoring.append(sim)
    solver     = SATSolver()
    solver     = solver(code, lambda: 'German')
    scoring.append(1.0 if solver else 0.0)

    return True, {'scores': scoring}

@toggle_test(True, default=MOCK_RETURN)
def test_solve_puzzle():
    scoring  = []
    reader   = FileReader()
    dir_path = Path(__file__).parent.absolute() / "snippets"

    problem        = reader((dir_path / "einstein_puzzle.txt").as_posix())
    solution       = reader((dir_path / "einstein_puzzle_human_solution.txt").as_posix())
    human_solution = reader((dir_path / "jays_brother_human_solution.txt").as_posix())
    trajectories   = reader((dir_path / "jays_brother_trajectories.txt").as_posix())

    query = "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother?"
    task  = """Using the provided solution to the Einstein's puzzle as a reference, your task is to implement the `solve_puzzle` function with the z3 package and solve the "Who is Jay's brother" puzzle.
    Only return the code of the `solve_puzzle` function between a ```python ... ``` code block.
    """

    template = f"""
    {task}

    Carefully read the following problem statement and learn how to associate the problem statement with the solution.
    {problem}

    Analyze the solution and learn how to associate the solution with the problem statement.
    {solution}

    Now, solve the puzzle in a similar fashion. Assume the solver `S` is given and the post-processing of the returned `Const` query is handled somewhere else.
    {query}
    """

    # Nicely deal with the markdown
    pp = CodeExtractPostProcessor()

    # Solve the puzzle
    cv     = Conversation(auto_print=False)
    res    = cv(template)
    code   = pp(res.value, None, tag="python")

    # Execute the code
    try:
        exec(
            code,
            globals()
        )

        # Tension…
        S = Solver()
        solve_puzzle = globals().get("solve_puzzle")
        solution     = solve_puzzle(S)
        validator    = S.check()

        # …and release!
        if validator == sat:
            model = S.model()
            answer = model[solution]

            if "John" in str(answer):
                # Attaboy!
                scoring.append(1.0)
            else:
                # No cigar
                scoring.append(0.0)
        else:
            scoring.append(0.0)
    except Exception as e:
        scoring.append(0.0)

    # How good?
    random     = Symbol(RANDOM_SEQUENCE)
    rand_score = human_solution.similarity(random, metric='cosine')
    base_score = human_solution.similarity(trajectories.split("\n\n\n"), metric="cosine").mean()
    score      = human_solution.similarity(code, normalize=normalize(base_score, rand_score), metric='cosine')
    scoring.append(score)

    return True, {'scores': scoring}

