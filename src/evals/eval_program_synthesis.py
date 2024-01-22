import os

from src.utils import normalize, rand_ast_measure, ast_measure, RANDOMNESS, MOCK_RETURN

from symai import Symbol, Expression, Conversation, Call
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
    # Define random sequence to normalize data
    random_seq = Symbol(RANDOMNESS).mean(axis=0)                                                                        | aggregate.random_seq
    # Create a template
    template   = os.path.join(cur_file_dir, 'snippets/latex_templating_problem.txt')
    conv       = Conversation(file_link=[template], auto_print=False)
    raw_res    = conv(task)                                                                                             | aggregate.gen_raw_res
    scoring    = []
    processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])
    code       = Symbol(processors(str(raw_res), None))                                                                 | aggregate.gen_code
    reader     = FileReader()
    solution1  = reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_1.txt'))                         | aggregate.solution1
    solution2  = reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_2.txt'))                         | aggregate.solution2
    solutions  = Symbol([solution1, solution2]).mean(axis=0)                                                            | aggregate.solutions
    base_score = solution1.measure(solution2)                                                                           | aggregate.conv_base_score
    # remove the chance of simply rephrasing the task description
    rand_score = solutions.measure(random_seq)                                                                          | aggregate.conv_rand_score
    score      = solutions.measure(raw_res, normalize=normalize(base_score, rand_score))                                | aggregate.conv_score
    scoring.append(score.value)

    # Read the source code from files
    solution1  = Symbol(solution1, callables=[Call('measure', ast_measure)])
    # compute again normalization score but this time for AST measure
    base_score = solution1.measure(solution2)                                                                           | aggregate.ast_base_score
    rand_score = (0.5*(rand_ast_measure(solution1) + rand_ast_measure(solution2)))                                      | aggregate.ast_rand_score
    score      = solution1.measure(code, normalize=normalize(base_score, rand_score))                                   | aggregate.ast_score
    scoring.append(score.value)

    # Execute the code
    code       = reader(template).str().replace('{TODO}', str(code))
    runner     = Execute(enclosure=True)
    success    = False
    try:
        res    = runner(code)
        # extract the output from the locals
        out    = Symbol(res['locals']['_output_'])                                                                      | aggregate.code_output
        ori    = reader(os.path.join(cur_file_dir, 'snippets/latex_templating_output.txt'))                             | aggregate.code_solution
        # no normalization is needed here since the output has to be an exact match
        score  = out.measure(ori)                                                                                       | aggregate.code_score
        scoring.append(score.value)
        success = True
    except Exception as e:
        score  = 0.0                                                                                                    | aggregate.code_score
        scoring.append(score)

    return success, {'scores': scoring}


class APIExecutor(Expression):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.builder     = APIBuilder()
        self.executor    = StackTraceRetryExecutor(retries=0) # disable retries
        self._verbose    = verbose
        self._request    = None
        self._code       = None
        self._result     = None

    @property
    def _runnable(self):
        return self.executor._runnable

    def forward(self, aggregate, request: Symbol, presets, **kwargs) -> Symbol:
        answer, refs, code, code2, rand = presets()
        self._request = self._to_symbol(request)
        if self._verbose: print('[REQUEST]', self._request)
        # Generate the code to implement the API call
        self._code    = self.builder(self._request)
        if self._verbose: print('[GENERATED_CODE]', self._code)
        base_score    = code.measure(code2)                                                                             | aggregate.base_score
        rand_score    = rand.measure(refs)                                                                              | aggregate.rand_score
        code_score    = code.measure(self._code, normalize=normalize(base_score, rand_score))                           | aggregate.code_score
        code_score    = code_score.value
        # Execute the code to define the 'run' function
        try:
            self._result  = self.executor(self._code, request=self._request)                                            | aggregate.output
            if self._verbose: print('[RESULT]:', self._result)
            web_score     = answer.measure(self._result)                                                                | aggregate.web_score
            web_score     = web_score.value
        except Exception as e:
            self._result  = str(e)
            web_score     = 0.0                                                                                         | aggregate.web_score
        self._value       = self._result
        return [code_score, web_score]


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_api_builder(aggregate):
    answer    = Symbol("Yannic Kilcher")                                                                                | aggregate.answer
    rand_seq  = Symbol(RANDOMNESS).mean(axis=0)                                                                         | aggregate.random_seq
    reader    = FileReader()
    website   = reader(os.path.join(cur_file_dir, 'snippets/code_api_builder_website_result.txt'))
    ref_code  = reader(os.path.join(cur_file_dir, 'snippets/code_api_builder.txt'))                                     | aggregate.ref_code
    ref_code2 = reader(os.path.join(cur_file_dir, 'snippets/code_api_builder2.txt'))                                    | aggregate.ref_code2
    refs      = Symbol([ref_code, ref_code2]).mean(axis=0)                                                              | aggregate.refs
    executor  = APIExecutor() # creates code on the fly and executes it
    scores    = executor(aggregate,
                         'Fetch data from URL https://www.ykilcher.com/ and use Function to extract the full name of the author.', # the request
                         lambda: (answer, refs, ref_code, ref_code2, rand_seq)) # interprets the instruction to generate a HTTP request
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
""")                                                                                                                    | aggregate.solution1
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
""")                                                                                                                    | aggregate.solution2
    solutions = Symbol([solution1, solution2]).mean(axis=0)                                                             | aggregate.solutions
    rand_seq  = Symbol(RANDOMNESS).mean(axis=0)                                                                         | aggregate.random_seq
    builder   = ExpressionBuilder()
    code      = builder("Create a query Expression that is initializes a Function with a prompt and processes a data Symbol based on the custom Function.")
    runner    = RuntimeExpression()
    scoring   = []
    try:
        expr  = runner(code)
        score = 1.0                                                                                                     | aggregate.code_score
        scoring.append(score)
        # initialize the expression with the prompt
        query = expr('extract the names from the text')
    except:
        score = 0.0                                                                                                     | aggregate.code_score
        scoring.append(score)
    base_score  = solution1.measure(solution2)                                                                          | aggregate.base_score
    rand_score  = solutions.measure(rand_seq)                                                                           | aggregate.rand_score
    score       = solution1.measure(code, normalize=normalize(base_score, rand_score))                                  | aggregate.code_score
    scoring.append(score.value)
    try:
        # run the expression on the data
        res   = query('Hello my name is Max and I am 20 years old.')                                                    | aggregate.query_res
        score = res.measure('Max')                                                                                      | aggregate.query_score
        scoring.append(score.value)
    except:
        score = 0.0                                                                                                     | aggregate.query_score
        scoring.append(score)
    return True, {'scores': scoring}
