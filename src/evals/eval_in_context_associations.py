import numpy as np

from symai import Symbol, Expression
from symai.utils import toggle_test

from src.utils import MOCK_RETURN, success_score


ACTIVE = True


# Define basic test functions
@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_basic_factual_prompt() -> bool:
    '''Sanity check test if the basic prompt works'''
    sym = Expression.prompt('''[Last Instruction]
Return only a number as an answer.
[Last Query]
Give the meaning of life a number, meaning that the answer to life, the universe and everything is:
[Answer]''')
    # sanity check if models are working
    # every model must pass this basic test
    res = '42' in str(sym)
    assert res, f'Failed to find 42 in {str(sym)}'
    return True, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_basic_factual_prompt_pi() -> bool:
    '''Sanity check test if the basic prompt works'''
    sym = Expression.prompt('''[Last Instruction]
Return only a number as an answer.
[Last Query]
Write the first 10 digits of Pi:
[Last Answer]''')
    # sanity check if models are working
    # every model must pass this basic test
    base = '3.1415926535'
    sim  = sym.similarity(base)
    return True, {'scores': [sim]}


# Define the test functions based on in-context learning associations and compositions
@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_add_and_equals() -> bool:
    '''Test if the addition operator between two number symbols works'''
    sym = Symbol(1) + Symbol(2)
    return sym.int() == 3, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_add_and_equals() -> bool:
    '''Test if the addition operator between a number symbol and linguistic number symbol works'''
    sym = Symbol(17) + 'two' # auto cast to Symbol
    return sym.int() == 19, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_add_and_equals() -> bool:
    '''Test if the addition operator between a large number symbol and linguistic number symbol works'''
    sym = 'two hundred and thirty four' + Symbol(7000) # auto cast to Symbol
    return sym.int() == 7234, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_check_pi() -> bool:
    '''Test if a fuzzy equality between pi string symbol and an number approximation symbol works'''
    sym = Symbol('pi') # semantic understanding of pi
    # test if pi is equal to 3.14159265... by approximating
    return sym == '3.14159265...', success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_check_pi_2() -> bool:
    '''Test if a fuzzy equality between np.pi number symbol and an number approximation symbol works'''
    sym = Symbol(np.pi) # has high floating point precision
    # test if pi is equal to 3.14159265... by approximating
    return sym == '3.14159265...', success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_sub_and_contains() -> bool:
    '''Test if a semantic subtraction operator between two symbols works'''
    # semantic understanding of subtraction
    base = 'Hello, I would like a cup of coffee.'
    res  = (Symbol('Hello, I would like a cup of tea.') - Symbol('tea')) + 'coffee'
    sim  = res.similarity(base)
    return True, {'scores': [sim]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_compare() -> bool:
    '''Test if a comparison operator between two number symbols works'''
    res = Symbol(10) > Symbol(5)
    return res, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_compare_2() -> bool:
    '''Test if a semantic comparison operator between two symbols works'''
    res = Symbol(10) > Symbol('fifteen thousand')
    return res == False, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_AND_logic():
    '''Test if logical AND can be used to combine two symbols'''
    base = 'The horn only sounds on Sundays and I hear the horn.'
    res  = Symbol('the horn only sounds on Sundays') & Symbol('I hear the horn')
    sim  = res.similarity(base)
    return True, {'scores': [sim]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_OR_logic():
    '''Test if logical OR can be used to combine two symbols'''
    base    = 'The cat has whiskers and a tail.'
    subject = 'cat'
    res = Symbol(f'The {subject} has whiskers.') | Symbol(f'The {subject} has a tail.')
    sim = np.maximum(res.similarity(base), res.similarity('True'))
    return True, {'scores': [sim]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_XOR_logic():
    '''Test if logical XOR can be used to combine two symbols'''
    base = 'It is unknown if the duck quacks or not.'
    res  = Symbol('The duck quacks.') ^ Symbol('The duck does not quack.')
    sim  = np.maximum(res.similarity(base), res.similarity('False'))
    return True, {'scores': [sim]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_insert_lshift():
    '''Test if information can be inserted into a symbol using the LSHIFT operator'''
    base = 'I love to eat apples and bananas'
    sym  = Symbol('I love to eat apples')
    res  = sym << 'and bananas'
    sim  = res.similarity(base)
    return True, {'scores': [sim]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_insert_rshift():
    '''Test if information can be inserted into a symbol using the RSHIFT operator'''
    base = 'I love to eat apples and bananas'
    sym  = Symbol('I love to eat apples')
    res  = 'and bananas' >> sym
    sim  = res.similarity(base)
    return True, {'scores': [sim]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_extract_information():
    '''Test if information can be extracted from a symbol using the EXTRACT operator'''
    sym  = Symbol('I have an iPhone from Apple. And it is not cheap. I love to eat bananas, mangos, and oranges. My hobbies are playing football and basketball.')
    res  = sym.extract('fruits')
    res  = str(res).lower().strip()
    cnt  = 0
    # check if the EXTRACT operator retains the 3 essential words
    assert 'bananas' in res, f'Failed to find bananas in {res}'
    cnt += 1
    assert 'mangos'  in res, f'Failed to find mangos in {res}'
    cnt += 1
    assert 'oranges' in res, f'Failed to find oranges in {res}'
    cnt += 1
    return True, {'scores': [cnt/3.0]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_extract_contextual_information():
    '''Test if number information can be extracted from a symbol using the EXTRACT operator'''
    sym = Symbol("""Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 7410 tokens (2988 in your prompt; 4422 for the completion). Please reduce your prompt; or completion length.",
                    param=None, code=None, http_status=400, request_id=None)]""")
    res = sym.extract('requested tokens').int()
    # check if the EXTRACT operator gets the correct number of tokens
    assert res == 7410, f'Failed to find 7410 in {str(res)}'
    return True, success_score


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_filter():
    '''Test if filtering information can be applied to a symbol using the FILTER operator'''
    sym  = Symbol('Physics, Sports, Mathematics, Music, Art, Theater, Writing')
    res  = sym.filter('science related subjects')
    res  = str(res).lower().strip()
    cnt  = 0
    # check if the FILTER operator retains the essential words
    assert 'physics' in res, f'Failed to find physics in {res}'
    cnt += 1
    assert 'mathematics' in res, f'Failed to find mathematics in {res}'
    cnt += 1
    assert 'music' in res, f'Failed to find music in {res}'
    cnt += 1
    assert 'art' not in res, f'Failed to remove art in {res}'
    cnt += 1
    assert 'theater' not in res, f'Failed to remove theater in {res}'
    cnt += 1
    assert 'writing' not in res, f'Failed to remove writing in {res}'
    cnt += 1
    assert 'sports' not in res, f'Failed to remove sports in {res}'
    cnt += 1
    return True, {'scores': [cnt/7.0]}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_clean():
    '''Test if cleaning information can be applied to a symbol using the CLEAN operator'''
    base = 'Hello World'
    sym  = Symbol('Hello *&&7amp;;; \t\t\t\nWorld')
    res  = sym.clean()
    # check if the CLEAN operator retains the 2 essential words
    sim  = res.similarity(base)
    return True, {'scores': [sim]}

