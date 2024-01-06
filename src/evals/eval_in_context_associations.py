import copy
import numpy as np

from symai import Symbol, Expression


success_score = {'scores': [1.0]}


# Define basic test functions
def test_basic_prompt() -> bool:
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
    return True, copy.deepcopy(success_score)


def test_basic_prompt_2() -> bool:
    '''Sanity check test if the basic prompt works'''
    sym = Expression.prompt('''[Last Instruction]
Return only a number as an answer.
[Last Query]
Write the first 10 digits of Pi:
[Last Answer]''')
    # sanity check if models are working
    # every model must pass this basic test
    res = '3.1415' in str(sym)
    assert res, f'Failed to find 3.1415 in {str(sym)}'
    return True, copy.deepcopy(success_score)


# Define the test functions based on in-context learning associations and compositions
def test_add_and_equals() -> bool:
    '''Test if the addition operator between two number symbols works'''
    sym = Symbol(1) + Symbol(2)
    return str(sym) == '3', copy.deepcopy(success_score)


def test_add_and_equals() -> bool:
    '''Test if the addition operator between a number symbol and linguistic number symbol works'''
    sym = Symbol(17) + 'two' # auto cast to Symbol
    return str(sym) == '19', copy.deepcopy(success_score)


def test_add_and_equals() -> bool:
    '''Test if the addition operator between a large number symbol and linguistic number symbol works'''
    sym = 'two hundred and thirty four' + Symbol(7000) # auto cast to Symbol
    return str(sym) == '7234', copy.deepcopy(success_score)


def test_check_pi() -> bool:
    '''Test if a fuzzy equality between pi string symbol and an number approximation symbol works'''
    sym = Symbol('pi') # semantic understanding of pi
    # test if pi is equal to 3.14159265... by approximating
    return sym == '3.14159265...', copy.deepcopy(success_score)


def test_check_pi_2() -> bool:
    '''Test if a fuzzy equality between np.pi number symbol and an number approximation symbol works'''
    sym = Symbol(np.pi) # has high floating point precision
    # test if pi is equal to 3.14159265... by approximating
    return sym == '3.14159265...', copy.deepcopy(success_score)


def test_sub_and_contains() -> bool:
    '''Test if a semantic subtraction operator between two symbols works'''
    # semantic understanding of subtraction
    res = (Symbol('Hello my friend.') - Symbol('friend')) + 'enemy'
    res = str(res).lower().strip()
    return ' enemy.' in res, copy.deepcopy(success_score)


def test_compare() -> bool:
    '''Test if a comparison operator between two number symbols works'''
    res = Symbol(10) > Symbol(5)
    res = str(res)
    return res == 'True', copy.deepcopy(success_score)


def test_compare_2() -> bool:
    '''Test if a semantic comparison operator between two symbols works'''
    res = Symbol(10) > Symbol('fifteen thousand')
    res = str(res)
    return res == 'False', copy.deepcopy(success_score)


def test_AND_logic():
    '''Test if logical AND can be used to combine two symbols'''
    res = Symbol('the horn only sounds on Sundays') & Symbol('I hear the horn')
    res = str(res).lower().strip()
    # check if the AND operator retains the 3 essential words
    val = 'sunday' in res
    assert val, f'Failed to find Sundays in {res}'
    val = 'horn' in res
    assert val, f'Failed to find horn in {res}'
    val = 'hear' in res
    assert val, f'Failed to find hear in {res}'
    return True, copy.deepcopy(success_score)


def test_OR_logic():
    '''Test if logical OR can be used to combine two symbols'''
    subject = 'cat'
    res = Symbol(f'The {subject} has whiskers.') | Symbol(f'The {subject} has a tail.')
    res = str(res).lower().strip()
    # check if the OR operator returns either a true value or it retains the subject
    ret = 'true' in str(res) or 'cat' in str(res)
    assert ret, f'Failed to find cat in {res}'
    return True, copy.deepcopy(success_score)


def test_XOR_logic():
    '''Test if logical XOR can be used to combine two symbols'''
    res = Symbol('The duck quaks.') ^ Symbol('The duck does not quak.')
    res = str(res).lower().strip()
    # check if the XOR operator returns False or it retains the main subject
    ret = 'false' in str(res) or ('duck' in str(res) and 'quack' in str(res))
    assert ret, f'Failed to find duck in {res}'
    return True, copy.deepcopy(success_score)


def test_insert_lshift():
    '''Test if information can be inserted into a symbol using the LSHIFT operator'''
    sym = Symbol('I love to eat apples')
    res = sym << 'and bananas'
    res = str(res).lower().strip()
    # check if the LSHIFT operator retains the 4 essential words
    assert 'love' in str(res), f'Failed to find love in {res}'
    assert 'apples and bananas' in str(res), f'Failed to find apples and bananas in {res}'
    assert 'eat' in str(res), f'Failed to find eat in {res}'
    return True, copy.deepcopy(success_score)


def test_insert_rshift():
    '''Test if information can be inserted into a symbol using the RSHIFT operator'''
    sym = Symbol('I love to eat apples')
    res = 'and bananas' >> sym
    res = str(res).lower().strip()
    # check if the RSHIFT operator retains the 4 essential words
    assert 'love' in res, f'Failed to find love in {res}'
    assert 'apples and bananas' in res, f'Failed to find apples and bananas in {res}'
    assert 'eat' in res, f'Failed to find eat in {res}'
    return True, copy.deepcopy(success_score)


def test_extract():
    '''Test if information can be extracted from a symbol using the EXTRACT operator'''
    sym = Symbol('I have an iPhone from Apple. And it is not cheap. I love to eat bananas, mangos, and oranges. My hobbies are playing football and basketball.')
    res = sym.extract('fruits')
    res = str(res).lower().strip()
    # check if the EXTRACT operator retains the 3 essential words
    assert 'bananas' in res, f'Failed to find bananas in {res}'
    assert 'mangos'  in res, f'Failed to find mangos in {res}'
    assert 'oranges' in res, f'Failed to find oranges in {res}'
    return True, copy.deepcopy(success_score)


def test_extract_tokens():
    '''Test if number information can be extracted from a symbol using the EXTRACT operator'''
    sym = Symbol("""Exception: Failed to query GPT-3 after 3 retries. Errors: [InvalidRequestError(message="This model's maximum context length is 4097 tokens, however you requested 7410 tokens (2988 in your prompt; 4422 for the completion). Please reduce your prompt; or completion length.",
                    param=None, code=None, http_status=400, request_id=None)]""")
    res = sym.extract('requested tokens').cast(int)
    # check if the EXTRACT operator gets the correct number of tokens
    assert res == 7410, f'Failed to find 7410 in {str(res)}'
    return True, copy.deepcopy(success_score)


def test_filter():
    '''Test if filtering information can be applied to a symbol using the FILTER operator'''
    sym = Symbol('Physics, Sports, Mathematics, Music, Art, Theater, Writing')
    res = sym.filter('science related subjects')
    res = str(res).lower().strip()
    # check if the FILTER operator retains the 3 essential words
    assert 'physics' in res, f'Failed to find physics in {res}'
    assert 'mathematics' in res, f'Failed to find mathematics in {res}'
    assert 'music' in res, f'Failed to find music in {res}'
    assert 'art' not in res, f'Failed to remove art in {res}'
    assert 'theater' not in res, f'Failed to remove theater in {res}'
    assert 'writing' not in res, f'Failed to remove writing in {res}'
    assert 'sports' not in res, f'Failed to remove sports in {res}'
    return True, copy.deepcopy(success_score)


def test_clean():
    '''Test if cleaning information can be applied to a symbol using the CLEAN operator'''
    sym = Symbol('Hello *&&7amp;;; \t\t\t\nWorld')
    res = sym.clean()
    res = str(res).lower().strip()
    # check if the CLEAN operator retains the 2 essential words
    assert 'hello world' == res, f'Failed to find hello world in {res}'
    return True, copy.deepcopy(success_score)

