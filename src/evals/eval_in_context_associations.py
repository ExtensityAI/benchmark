import copy
from symai import Symbol


success_score = {'score': 1.0}


# Define the test functions based on in-context learning associations
def test_add_and_equals() -> bool:
    sym = Symbol(1) + Symbol('two')
    return str(sym) == '3', copy.deepcopy(success_score)

def test_sub_and_contains() -> bool:
    res = (Symbol('Hello my friend.') - Symbol('friend')) + 'enemy'
    res = str(res).lower().strip()
    return 'enemy' in res and res.endswith('.'), copy.deepcopy(success_score)

def test_compare() -> bool:
    res = Symbol(10) > Symbol(5)
    res = str(res)
    return res == 'True', copy.deepcopy(success_score)

