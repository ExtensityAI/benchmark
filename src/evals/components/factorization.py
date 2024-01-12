from symai import Function


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
