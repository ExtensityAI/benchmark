from symai import Symbol


# Define the test functions based on in-context learning associations
def test_add_and_equals() -> bool:
    sym = Symbol(1) + Symbol(2)
    return sym == 3

def test_sub_and_contains() -> bool:
    res = (Symbol('Hello my friend') - Symbol('friend')) + 'enemy'
    return 'enemy' in res

def test_compare() -> bool:
    res = Symbol(10) > Symbol(5)
    return res

