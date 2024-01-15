# good reference: https://ericpony.github.io/z3py-tutorial/guide-examples.htm

from symai import Symbol, Expression
from symai.components import FileReader, Execute, Function
from symai.extended import Conversation

from z3 import Solver, sat


def register_custom_engine():
    from symai.functional import EngineRepository
    from engine import LLaMACppClientEngine

    ENGINE = "zephyr"
    HOST   = "http://localhost"
    PORT   = 8081
    assert ENGINE in ["zephyr", "mistral"]

    engine = LLaMACppClientEngine(host=HOST, port=PORT)
    EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)


# tokens: 149
problem = Symbol("""
The Englishman lives in the red house.
The Swede keeps dogs.
The Dane drinks tea.
The green house is just to the left of the white one.
The owner of the green house drinks coffee.
The Pall Mall smoker keeps birds.
The owner of the yellow house smokes Dunhills.
The man in the center house drinks milk.
The Norwegian lives in the first house.
The Blend smoker has a neighbor who keeps cats.
The man who smokes Blue Masters drinks bier.
The man who keeps horses lives next to the Dunhill smoker.
The German smokes Prince.
The Norwegian lives next to the blue house.
The Blend smoker has a neighbor who drinks water.
The question to be answered is: Who keeps fish?
""")

reader = FileReader()
runner = Execute(enclosure=True)

# tokens: 1383
solution = reader("einstein_puzzle_solution.txt")

# tokens: 38
query = Symbol("Bob has two sons, John and Jay. Jay has one brother and father. The father has two sons. Jay's brother has a brother and a father. Who is Jay's brother.")
task = "Using the provided solution to the Einstein puzzle as a reference, apply the Z3 solver to solve the puzzle. Provide executable code that upon running is solving the puzzle."

# total: 1571
template = f"""
# Problem
{problem.value}

# Solution
{solution.value}

# Problem
{query.value}

# Task
{task}
"""


cv = Conversation()
res = cv(template)
is_mistake = res.query("Is there a mistake in this program? Write `No` if there are no mistakes, or the line where there is a mistake.", temperature=0)
breakpoint()


