import os
import numpy as np

from src.utils import normalize, ast_similarity

from symai import Symbol, Conversation
from symai.components import FileReader
from symai.processor import ProcessorPipeline
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor


cur_file_dir = os.path.dirname(os.path.abspath(__file__))


def test_latex_templating():
    # Create a template
    conv = Conversation(file_link=[os.path.join(cur_file_dir, 'snippets/latex_templating_problem.txt')], auto_print=False)
    res  = conv("""[Task]
Create a function `create_latex_result` that takes in the `benchmark_results` as `data` and parses the LaTeX table rows and columns based on the `data` results. The table should follow the `latex_template` format and populate the rows table as indicated by the placeholder variables. Mark the best performing model per row with bold text. At the bottom of the benchmarks, place the values of the total row by computing the average over all columns and populating the `total_values` entry in the `latex_template`.
The table should be returned as a string by the function.
All required imports are already provided. The code of the `create_latex_result` function should be written between a
```python
...
```
code block.
The `create_latex_result` function must be self-contained, fully functional and pass all tests.
No other functions or explanations are required.
""")
    scores     = []
    processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])
    code       = Symbol(processors(str(res), None))
    reader     = FileReader()
    solution1  = Symbol(reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_1.txt')))
    solution2  = Symbol(reader(os.path.join(cur_file_dir, 'snippets/latex_templating_solution_2.txt')))
    norm_score = solution1.similarity(solution2)
    score      = np.minimum(score, normalize=normalize(norm_score))
    scores.append(score)

    # Read the source code from files
    file1 = os.path.join(cur_file_dir, 'snippets/latex_templating_solution_1.txt')
    file2 = os.path.join(cur_file_dir, 'snippets/latex_templating_solution_2.txt')
    file1 = Symbol(file1, callables={'similarity': ast_similarity})
    norm_score = file1.similarity(file2)
    score      = file1.similarity(code, normalize=normalize(norm_score))
