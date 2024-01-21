from pathlib import Path

import sympy as sym
from symai import Conversation, Function, Symbol
from symai.components import FileReader
from symai.extended import Conversation
from symai.post_processors import CodeExtractPostProcessor, StripPostProcessor
from symai.utils import toggle_test

from src.evals.components import Factorization
from src.utils import MOCK_RETURN, RANDOM_SEQUENCE, REVERSED_RANDOM_SEQUENCE, normalize

from z3 import Solver, sat

ACTIVE = True


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
def test_factorize_formula(aggregate):
    a, b, c, d, x, y = sym.symbols('a, b, c, d, x, y')
    expr        = a * x + b * x - c * x - a * y - b * y + c * y + d
    stmt        = Symbol("Can you simplify me the following expression: a*x + b*x - c*x - a*y - b*y + c*y + d")
    random_seq  = Symbol([RANDOM_SEQUENCE, REVERSED_RANDOM_SEQUENCE]).mean(axis=0)                                         | aggregate.random_seq
    #res goes to sympy
    symbols_    = stmt.extract('all unique symbols as a list')
    refs        = Symbol(['a, b, c, d, x, y',
                          'y, x, a, b, c, d',
                          'a, b, c, d, x, y',
                          'a, b, c, d, x, y',
                          'a, b, c, d, x, y',
                          'a, b, c, d, x, y'])
    mean_refs   = refs.mean()                                                                                              | aggregate.symbols_mean_refs
    base_score  = refs.cvs()                                                                                               | aggregate.symbols_base_score
    rand_score  = random_seq.measure(mean_refs)                                                                            | aggregate.symbols_rand_score
    score       = symbols_.measure(mean_refs, normalize=normalize(base_score, rand_score))                                 | aggregate.symbols_score
    # validate with sympy
    fact        = sym.collect(expr, d, func=sym.factor)
    # model based factorization
    func        = Factorization('Factorize d from the expression such that your final start with: `d + (...`:')
    res         = func(expr)                                                                                               | aggregate.generated
    ref         = Symbol(fact)                                                                                             | aggregate.solution
    rand_score  = ref.measure(random_seq)
    solutions   = Symbol(["The factorized result is: d+(a+b-c)*(x-y)",
                          "We obtain: d + ( x - y ) * ( a + b - c )",
                          "(a + b - c) * (x - y) + d"])
    sol_mean    = solutions.mean()                                                                                         | aggregate.solution_mean
    base_score  = solutions.cvs()                                                                                          | aggregate.solution_base_score
    # validate
    score       = ref.measure(res, normalize=normalize(base_score, rand_score))                                            | aggregate.solution_score
    return True, {'scores': [score]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_dsl_writing_capability(aggregate):
    # test only the capability to follow template instructions from a custom DSL (syntax) + semantic structure
    val  = "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother."
    reader       = FileReader()
    dir_path     = Path(__file__).parent.absolute() / "snippets"
    formulations = reader((dir_path / "jays_brother_trajectories.txt").as_posix())
    formulation1, formulation2 = formulations.split("\n\n\n")
    scoring      = []
    expr         = HOLFactorization(val, post_processors=[StripPostProcessor(), CodeExtractPostProcessor()])
    res          = expr(val)                                                                                               | aggregate.generated
    form1        = Symbol(formulation1)                                                                                    | aggregate.solution1
    form2        = Symbol(formulation2)                                                                                    | aggregate.solution2
    random       = Symbol(RANDOM_SEQUENCE) # remove the chance of simply rephrasing the question
    rand_score   = random.measure([form1, form2]).mean()                                                                   | aggregate.rand_score
    base_score   = form1.measure(form2)                                                                                    | aggregate.base_score
    score        = form1.measure(res, normalize=normalize(base_score, rand_score))                                         | aggregate.score
    scoring.append(score)
    # vary basic check for syntax violations
    if '("' in str(res) or '")' in str(res) or '",' in str(res) or '":' in str(res) or '=' in str(res):
        score = 0.0                                                                                                        | aggregate.score
        scoring.append(score)
        return False, {'scores': scoring}
    else:
        score = 1.0                                                                                                        | aggregate.score
        scoring.append(score)
        return True, {'scores': scoring}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_solve_puzzle(aggregate):
    scoring  = []
    reader   = FileReader()
    dir_path = Path(__file__).parent.absolute() / "snippets"

    problem        = reader((dir_path / "einstein_puzzle.txt").as_posix())
    solution       = reader((dir_path / "einstein_puzzle_human_solution.txt").as_posix())
    ref_solution   = reader((dir_path / "jays_brother_human_solution.txt").as_posix())                                     | aggregate.ref_solution
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
    res    = cv(template)                                                                                                  | aggregate.gen_raw_res
    code   = pp(res.value, None, tag="python")                                                                             | aggregate.gen_code
    succ   = False

    # Execute the code
    try:
        exec(
            code,
            globals()
        )
        # see also Expression: symai.extended import SATSolver

        # Tension…
        S = Solver()
        solve_puzzle = globals().get("solve_puzzle")
        solution     = solve_puzzle(S)
        validator    = S.check()
        # Some reward for being capable of writing executable code…
        score        = 1.0                                                                                                | aggregate.score
        scoring.append(score)

        # …but did you get it right?
        if validator == sat:
            model  = S.model()
            answer = model[solution]
            # CHECK: if Attaboy! or No cigar
            score  = 1.0 if "John" in str(answer) else 0.0                                                                | aggregate.score
            scoring.append(score)
            succ = True # at least runnable
        else:
            scoring.append(0.0)                                                                                           | aggregate.score
    except Exception as e:
        # not runnable
        scoring.append(0.0)                                                                                               | aggregate.score
        # not verifiable
        scoring.append(0.0)                                                                                               | aggregate.score

    # How good?
    random     = Symbol([RANDOM_SEQUENCE, REVERSED_RANDOM_SEQUENCE]).mean(axis=0)                                         | aggregate.random_seq
    rand_score = ref_solution.measure(random)                                                                             | aggregate.rand_score
    solutions  = trajectories.split("\n\n\n").mean()                                                                      | aggregate.solutions
    base_score = ref_solution.measure(solutions)                                                                          | aggregate.base_score
    score      = ref_solution.measure(res, normalize=normalize(base_score, rand_score))                                   | aggregate.gen_score
    scoring.append(score)

    return succ, {'scores': scoring}
