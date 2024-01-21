import os

from src.utils import normalize, rand_ast_measure, ast_measure, RANDOM_SEQUENCE, MOCK_RETURN

from symai import Symbol, Expression, Conversation
from symai.components import FileReader, Execute, RuntimeExpression, ExpressionBuilder
from symai.processor import ProcessorPipeline
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor
from symai.utils import toggle_test
from symai.extended.api_builder import APIBuilder, StackTraceRetryExecutor


ACTIVE = True
cur_file_dir = os.path.dirname(os.path.abspath(__file__))


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_application_template(aggregate):
    task      = """[Task]
Create a function `create_latex_result` that takes in the `benchmark_results` as `data` and parses the LaTeX table rows and columns based on the `data` results. The table should follow the `latex_template` format and populate the rows table as indicated by the placeholder variables. Mark the best performing model per row with bold text. At the bottom of the benchmarks, place the values of the total row by computing the average over all columns and populating the `total_values` entry in the `latex_template`.
The table should be returned as a string by the function.
All required imports are already provided. The code of the `create_latex_result` function should be written between a
```python
...
```
code block.
The `create_latex_result` function must be self-contained, fully functional and pass all tests.
No other functions or explanations are required.
"""
    # Create a template
    template   = os.path.join(cur_file_dir, 'snippets/latex_templating_problem.txt')
    conv       = Conversation(file_link=[template], auto_print=False)
    raw_res    = conv(task)
    scoring    = []
    processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])
    code       = Symbol(processors(str(raw_res), None))
    reader     = FileReader()
    solution1  = reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_1.txt'))
    solution2  = reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_2.txt'))
    base_score = solution1.measure(solution2)
    random_seq = RANDOM_SEQUENCE # remove the chance of parsing sub-sequence from the task description
    rand_score = solution1.measure(random_seq) # remove the chance of simply rephrasing the task
    score      = solution1.measure(raw_res, normalize=normalize(base_score, rand_score))
    scoring.append(score)

    # Read the source code from files
    solution1  = Symbol(solution1, callables={'measure': ast_measure})
    # compute again normalization score but this time for AST measure
    base_score = solution1.measure(solution2)
    rand_score = 0.5*(rand_ast_measure(solution1, random_seq) + rand_ast_measure(solution2, random_seq))
    score      = solution1.measure(code, normalize=normalize(base_score, rand_score))
    scoring.append(score)

    # Execute the code
    code       = reader(template).str().replace('{TODO}', str(code))
    runner     = Execute(enclosure=True)
    success    = False
    try:
        res    = runner(code)
        # extract the output from the locals
        out    = Symbol(res['locals']['_output_'])
        ori    = reader(os.path.join(cur_file_dir, 'snippets/latex_templating_output.txt'))
        # no normalization is needed here since the output has to be an exact match
        score  = out.measure(ori)
        scoring.append(score)
        success = True
    except Exception as e:
        scoring.append(0.0)

    return success, {'scores': scoring}


class APIExecutor(Expression):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.builder   = APIBuilder()
        self.executor  = StackTraceRetryExecutor(retries=0) # disable retries
        self._verbose  = verbose
        self._request  = None
        self._code     = None
        self._result   = None
        self._code_sim = None

    @property
    def _runnable(self):
        return self.executor._runnable

    def forward(self, request: Symbol, presets, **kwargs) -> Symbol:
        ref, code, code2, rand = presets()
        self._request = self._to_symbol(request)
        if self._verbose: print('[REQUEST]', self._request)
        # Generate the code to implement the API call
        self._code    = self.builder(self._request)
        if self._verbose: print('[GENERATED_CODE]', self._code)
        base_sim      = code.measure(code2)
        rand_sim      = rand.measure(code2)
        code_sim      = code.measure(self._code, normalize=normalize(base_sim, rand_sim))
        # Execute the code to define the 'run' function
        try:
            self._result  = self.executor(self._code, request=self._request)
            if self._verbose: print('[RESULT]:', self._result)
            web_sim       = ref.measure(self._result)
        except Exception as e:
            self._result  = str(e)
            web_sim       = 0.0
        self._value       = self._result
        return [code_sim, web_sim]


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_api_builder(aggregate):
    ref       = Symbol("Yannic Kilcher")
    rand_seq  = Symbol(RANDOM_SEQUENCE)
    reader    = FileReader()
    website   = reader(os.path.join(cur_file_dir, 'snippets/code_api_builder_website_result.txt'))
    ref_code  = reader(os.path.join(cur_file_dir, 'snippets/code_api_builder.txt'))
    ref_code2 = reader(os.path.join(cur_file_dir, 'snippets/code_api_builder2.txt'))
    source    = APIExecutor() # creates code on the fly and executes it
    scores    = source('Fetch data from URL https://www.ykilcher.com/ and use Function to extract the full name of the author.', # the request
                      lambda: (ref, ref_code, ref_code2, rand_seq)) # interprets the instruction to generate a HTTP request
    return True, {'scores': scores}


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_expression_builder(aggregate):
    solution1 = Symbol("""
# do not remove or change the imports
from symai import Expression, Function, Symbol
class QueryExpression(Expression):
    # initialize the expression with task specific arguments
    def __init__(self, prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.func = Function(prompt, **kwargs)

    # define the forward function with data specific arguments
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        result = self.func(sym, *args, **kwargs)
        return result
# assign the expression type to the variable _value_obj_
_value_obj_ = QueryExpression
""")
    solution2 = Symbol("""
from symai import Expression, Function, Symbol
class QueryExpression(Expression):
    def __init__(self, prompt: str, **kwargs):
        super().__init__(**kwargs)
        self.func = Function(prompt, **kwargs)
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym)
        return self.func(sym, *args, **kwargs)
_value_obj_ = QueryExpression
""")
    builder  = ExpressionBuilder()
    code     = builder("Create a query Expression that is initializes a Function with a prompt and processes a data Symbol based on the custom Function.")
    runner   = RuntimeExpression()
    scoring  = []
    try:
        expr    = runner(code)
        scoring.append(1.0)
    except:
        scoring.append(0.0)
    query    = expr('extract the names from the text')
    base_sim = solution1.measure(solution2)
    rand_sim = solution1.measure(RANDOM_SEQUENCE)
    sim      = solution1.measure(code, normalize=normalize(base_sim, rand_sim))
    scoring.append(sim)
    try:
        res  = query('Hello my name is Max and I am 20 years old.')
        sim  = res.measure('Max')
        scoring.append(sim)
    except:
        scoring.append(0.0)
    return True, {'scores': scoring}
