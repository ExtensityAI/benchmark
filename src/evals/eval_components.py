import sympy as sym

from symai import Symbol, Expression, Function
from symai.utils import toggle_test

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


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_causal_expression():
    val  = "Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother."
    expr = Factorization(val)
    res  = expr(val)
    assert res == '4', f'Failed to find 4 in {str(res)}'
    return True, success_score
