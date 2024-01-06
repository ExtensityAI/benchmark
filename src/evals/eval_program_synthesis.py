import os

from src.utils import normalize, rand_ast_similarity, ast_similarity, RANDOM_SEQUENCE, MOCK_RETURN

from symai import Symbol, Conversation
from symai.components import FileReader, Execute
from symai.processor import ProcessorPipeline
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor
from symai.utils import toggle_test


ACTIVE = True
cur_file_dir = os.path.dirname(os.path.abspath(__file__))


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_latex_templating():
    task     = """[Task]
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
    template = os.path.join(cur_file_dir, 'snippets/latex_templating_problem.txt')
    conv     = Conversation(file_link=[template], auto_print=False)
    res      = conv(task)
    scoring    = []
    processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])
    code       = Symbol(processors(str(res), None))
    reader     = FileReader()
    solution1  = Symbol(reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_1.txt')))
    solution2  = Symbol(reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_2.txt')))
    base_score = solution1.similarity(solution2)
    rand_score = solution1.similarity(RANDOM_SEQUENCE+task) # remove the chance of simply rephrasing the task
    score      = code.similarity(solution1, normalize=normalize(base_score, rand_score))
    scoring.append(score)

    # Read the source code from files
    solution1  = Symbol(solution1, callables={'similarity': ast_similarity})
    # compute again normalization score but this time for AST similarity
    base_score = solution1.similarity(solution2)
    rand_score = 0.5*(rand_ast_similarity(solution1) + rand_ast_similarity(solution2))
    score      = solution1.similarity(code, normalize=normalize(base_score, rand_score))
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
        score  = out.similarity(ori)
        scoring.append(score)
        success = True
    except Exception as e:
        scoring.append(0.0)

    return success, {'scores': scoring}
