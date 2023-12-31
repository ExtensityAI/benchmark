import copy
from symai import Symbol


success_score = {'score': 1.0}


# Define the test functions based on in-context learning associations
def test_add_and_equals() -> bool:
    sym = Symbol(1) + Symbol('two')
    return str(sym) == '3', copy.deepcopy(success_score)

def test_sub_and_contains() -> bool:
    res = (Symbol('Hello my friend') - Symbol('friend')) + 'enemy'
    return 'enemy' in str(res).lower(), copy.deepcopy(success_score)

def test_compare() -> bool:
    res = Symbol(10) > Symbol(5)
    return res, copy.deepcopy(success_score)

