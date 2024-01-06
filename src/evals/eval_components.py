import sympy as sym

from symai import Symbol, Expression, Function
from symai.utils import toggle_test
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor

from src.utils import normalize, RANDOM_SEQUENCE, MOCK_RETURN, success_score


ACTIVE = False


FACTORIZATION_CONTEXT = """[Context]
Compute the factorization of expression, ``f``, into irreducibles. (To
factor an integer into primes, use ``factorint``.)

There two modes implemented: symbolic and formal. If ``f`` is not an
instance of :class:`Poly` and generators are not specified, then the
former mode is used. Otherwise, the formal mode is used.

In symbolic mode, :func:`factor` will traverse the expression tree and
factor its components without any prior expansion, unless an instance
of :class:`~.Add` is encountered (in this case formal factorization is
used). This way :func:`factor` can handle large or symbolic exponents.

By default, the factorization is computed over the rationals. To factor
over other domain, e.g. an algebraic or finite field, use appropriate
options: ``extension``, ``modulus`` or ``domain``.
"""


class Factorization(Function):
    @property
    def static_context(self):
        return FACTORIZATION_CONTEXT


LOGIC_FACTORIZATION_CONTEXT = """[Context]
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


class LogicFactorization(Function):
    @property
    def static_context(self):
        return LOGIC_FACTORIZATION_CONTEXT


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_factorize_formula():
    a, b, c, d, x, y = sym.symbols('a, b, c, d, x, y')
    expr        = a * x + b * x - c * x - a * y - b * y + c * y + d
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
    # (score - rand) / (baseline - rand) =
    # score / (baseline - rand) - (rand / (baseline - rand)) =
    # score * z, z = 1 / (baseline - rand) - (rand / (baseline - rand)) =
    # score * ((1 - rand) / (baseline - rand))
    score       = ref.similarity(res, normalize=normalize(base_score, rand_score))

    return True, {'scores': [score]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_linear_function_composition():
    val  = "A line parallel to y = 4x + 6 passes through a point P=(x1=5, y1=10). What is the y-coordinate of the point where this line crosses the y-axis?"
    expr = Factorization('Rewrite the equation in the form y = mx + b and solve the problem.')
    res  = expr(val)
    assert '-10' in str(res), f'Failed to find 6 in {str(res)}'
    return True, success_score


#@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_causal_expression():
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
    expr        = LogicFactorization(val, post_processors=[StripPostProcessor(), CodeExtractPostProcessor()])
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

