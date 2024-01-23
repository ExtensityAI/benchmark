import numpy as np

from typing import Any

from symai import core
from symai import Symbol, Expression
from symai.utils import toggle_test
from symai.ops.primitives import Primitive

from src.utils import MOCK_RETURN, RANDOMNESS, bool_success, normalize


ACTIVE = True


# Define basic test functions
@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_basic_factual_prompt(aggregate):
    '''Sanity check test if the basic prompt works'''
    sym = Expression.prompt('''[Last Instruction]
Return only a number as an answer.
[Last Query]
Give the meaning of life a number, meaning that the answer to life, the universe and everything is:
[Answer]''')
    # sanity check if models are working
    # every model must pass this basic test
    res = ('42' in str(sym))                             | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_basic_factual_prompt_pi(aggregate):
    '''Sanity check test if the basic prompt works'''
    sym = Expression.prompt('''[Last Instruction]
Return only a number as an answer.
[Last Query]
Write the first 10 digits of Pi:
[Last Answer]''')                                        | aggregate.sym         # collect the symbol value
    # sanity check if models are working
    # every model must pass this basic test
    base  = Symbol('3.1415926535')                       | aggregate.base        # collect the base value
    score = sym.measure(base)                            | aggregate.score       # collect the score
    return True, {'scores': [score.value]}


# Define the test functions based on in-context learning associations and compositions
@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_add_and_equals(aggregate):
    '''Test if the addition operator between two number symbols works'''
    try:
        sym = (Symbol(1) + Symbol(2)).int()
    except:
        sym = 0 # default value for failure
    res = (sym == 3)                                     | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_add_and_equals(aggregate):
    '''Test if the addition operator between a number symbol and linguistic number symbol works'''
    # auto cast to Symbol
    try:
        sym = (Symbol(17) + 'two').int()
    except:
        sym = 0 # default value for failure
    res = (sym == 19)                                    | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_add_and_equals(aggregate):
    '''Test if the addition operator between a large number symbol and linguistic number symbol works'''
    # auto cast to Symbol
    try:
        sym = ('two hundred and thirty four' + Symbol(7000)).int()
    except:
        sym = 0 # default value for failure
    res = (sym == 7234)                                  | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_check_pi(aggregate):
    '''Test if a fuzzy equality between pi string symbol and an number approximation symbol works'''
    # semantic understanding of pi
    sym = Symbol('pi')                                   | aggregate.sym         # collect the symbol value
    # test if pi is equal to 3.14159265... by approximating
    res = (sym == '3.14159265...')                       | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_check_pi_2(aggregate):
    '''Test if a fuzzy equality between np.pi number symbol and an number approximation symbol works'''
    # has high floating point precision
    sym = Symbol(np.pi)                                  | aggregate.sym         # collect the symbol value
    # test if pi is equal to 3.14159265... by approximating
    res = (sym == '3.14159265...')                       | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_sub_and_contains(aggregate):
    '''Test if a semantic subtraction operator between two symbols works'''
    # semantic understanding of subtraction
    base  = 'Hello, I would like a cup of coffee.'                                        | aggregate.base       # collect the base value
    res   = ((Symbol('Hello, I would like a cup of tea.') - Symbol('tea')) + 'coffee')    | aggregate.res        # collect the result value
    score = res.measure(base)                                                             | aggregate.score      # collect the score
    return True, {'scores': [score.value]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_compare(aggregate):
    '''Test if a comparison operator between two number symbols works'''
    res = (Symbol(10) > Symbol('5'))
    res = (res == True)                                                                   | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_compare_2(aggregate):
    '''Test if a semantic comparison operator between two symbols works'''
    res = Symbol(10) < Symbol('fifteen thousand')
    res = (res == True)                                                                   | aggregate.res         # collect the result value
    return res, bool_success(res)


class CustomLogicPrimitive(Primitive):
    def __or__(self, other: Any) -> Any:
        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass # could impl. a fallback behavior here
        return self._to_symbol(_func(self, other))

    def __ror__(self, other: Any) -> Any:
        @core.logic(operator='or')
        def _func(_, a: str, b: str):
            pass # could impl. a fallback behavior here
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self))

    def __and__(self, other: Any) -> Any:
        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass # could impl. a fallback behavior here
        return self._to_symbol(_func(self, other))

    def __rand__(self, other: Any) -> Any:
        @core.logic(operator='and')
        def _func(_, a: str, b: str):
            pass # could impl. a fallback behavior here
        other = self._to_symbol(other)
        return self._to_symbol(_func(other, self))


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_AND_logic(aggregate):
    '''Test if logical AND can be used to combine two symbols'''
    base       = Symbol(['The horn only sounds on Sundays and I hear the horn.',
                         'Since I hear the horn it is Sunday'])
    base_mean  = base.mean()                                                                      | aggregate.base_mean   # collect the mean base value
    base_score = base.cvs()                                                                       | aggregate.base_score  # collect the base value
    res        = (Symbol('the horn only sounds on Sundays', primitives=CustomLogicPrimitive) & \
                  Symbol('I hear the horn', primitives=CustomLogicPrimitive))                     | aggregate.res         # collect the result value
    rand_mean  = Symbol(RANDOMNESS).mean()                                                        | aggregate.rand_mean   # collect the mean random value
    rand_score = base_mean.measure(rand_mean)                                                     | aggregate.rand_score  # collect the random score
    score      = Symbol(res).measure(base_mean, normalize=normalize(base_score, rand_score))      | aggregate.score       # collect the score
    return True, {'scores': [score.value]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_OR_logic(aggregate):
    '''Test if logical OR can be used to combine two symbols'''
    base    = Symbol(['The cat has whiskers and a tail.',
                      'The cat has both, whiskers and a tail',
                      'The cat has both, a tail and whiskers'])
    base_mean  = base.mean()                                                                      | aggregate.base_mean   # collect the mean base value
    base_score = base.cvs()                                                                       | aggregate.base_score
    subject    = 'cat'
    res = (Symbol(f'The {subject} has whiskers.', primitives=CustomLogicPrimitive) | \
           Symbol(f'The {subject} has a tail.', primitives=CustomLogicPrimitive))                 | aggregate.res         # collect the result value
    rand_mean  = Symbol(RANDOMNESS).mean()                                                        | aggregate.rand_mean   # collect the mean random value
    rand_score = base_mean.measure(rand_mean)                                                     | aggregate.rand_score  # collect the random score
    score      = Symbol(res).measure(base_mean, normalize=normalize(base_score, rand_score))      | aggregate.score       # collect the score
    return True, {'scores': [score.value]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_XOR_logic(aggregate):
    '''Test if logical XOR can be used to combine two symbols'''
    base = Symbol(['It is unknown if the duck quacks or not.',
                   'This is a contradiction because the duck quacks and does not quack.',
                   'False, it is known if the duck quacks or not.'])
    base_mean  = base.mean()                                                                      | aggregate.base_mean   # collect the mean base value
    base_score = base.cvs()                                                                       | aggregate.base_score  # collect the base value
    res  = (Symbol('The duck quacks.') ^ Symbol('The duck does not quack.'))                      | aggregate.res         # collect the result value
    rand_mean  = Symbol(RANDOMNESS).mean()                                                        | aggregate.rand_mean   # collect the mean random value
    rand_score = base_mean.measure(rand_mean)                                                     | aggregate.rand_score  # collect the random score
    score      = res.measure(base_mean, normalize=normalize(base_score, rand_score))              | aggregate.score       # collect the score
    return True, {'scores': [score.value]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_insert_rshift(aggregate):
    '''Test if information can be inserted into a symbol using the RSHIFT operator'''
    base  = 'I love to eat apples and bananas'                                            | aggregate.base        # collect the base value
    sym   = Symbol('I love to eat apples')                                                | aggregate.sym         # collect the symbol value
    res   = ('and bananas' >> sym)                                                        | aggregate.res         # collect the result value
    # expect exact solution
    score = res.measure(base)                                                             | aggregate.score       # collect the score
    return True, {'scores': [score.value]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_extract_information(aggregate):
    '''Test if information can be extracted from a symbol using the EXTRACT operator'''
    sym  = Symbol('I have an iPhone from Apple. And it is not cheap. ' + \
                  'I love to eat bananas, mangos, and oranges. ' + \
                  'My hobbies are playing football and basketball.')                      | aggregate.sym         # collect the symbol value
    res  = sym.extract('fruits')
    res  = str(res).lower().strip()                                                       | aggregate.res         # collect the result value
    cnt  = 0
    succ = True
    # check if the EXTRACT operator retains the 3 essential words
    succ &= 'bananas' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'mangos' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'oranges' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    return succ, {'scores': [cnt/3.0]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_extract_contextual_information(aggregate):
    '''Test if number information can be extracted from a symbol using the EXTRACT operator'''
    sym = Symbol("""Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 7410 tokens (2988 in your prompt; 4422 for the completion). Please reduce your prompt; or completion length.",
                    param=None, code=None, http_status=400, request_id=None)]""")         | aggregate.sym         # collect the symbol value
    try:
        res = sym.extract('requested tokens').int() # cast to int
    except:
        res = 0 # default value
    # check if the EXTRACT operator gets the correct number of tokens
    res = (res == 7410)                                                                   | aggregate.res         # collect the result value
    return res, bool_success(res)


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_filter(aggregate):
    '''Test if filtering information can be applied to a symbol using the FILTER operator'''
    sym  = Symbol('Physics, Sports, Mathematics, Music, Art, Theater, Writing')           | aggregate.sym         # collect the symbol value
    res  = sym.filter('science related subjects')
    res  = str(res).lower().strip()                                                       | aggregate.res         # collect the result value
    cnt  = 0
    succ = True
    # check if the FILTER operator retains the essential words
    succ &= 'physics' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'mathematics' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'music' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'art' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'theater' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'writing' in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    succ &= 'sports' not in res
    cnt += (1 if succ else 0)                                                             | aggregate.cnt         # collect the result value
    return succ, {'scores': [cnt/7.0]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_clean(aggregate):
    '''Test if cleaning information can be applied to a symbol using the CLEAN operator'''
    base  = 'Hello World'                                                                 | aggregate.base        # collect the base value
    sym   = Symbol('Hello *&&7amp;;; \t\t\t\nWorld')                                      | aggregate.sym         # collect the symbol value
    res   = sym.clean()                                                                   | aggregate.res         # collect the result value
    # check if the CLEAN operator retains the 2 essential words
    # expect exact solution
    score = res.measure(base)                                                             | aggregate.score       # collect the score
    return True, {'scores': [score.value]}
