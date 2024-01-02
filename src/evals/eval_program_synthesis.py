import copy
from symai import Symbol, Conversation


success_score = {'score': 1.0}



def test_latex_templating():
    # Create a template
    conv = Conversation(file_link=['snippets/latex_templating.py'])
    conv("""[Task]
Create a function `create_latex_result` that takes in the `benchmark_results` as `data` and parses the LaTeX table rows and columns based on the `data` results. The table should follow the `latex_template` format and populate the rows table as indicated by the placeholder variables. Mark the best performing model per row with bold text. At the bottom of the benchmarks, place the values of the total row by computing the average over all columns and populating the `total_values` entry in the `latex_template`.
The table should be saved as `benchmark_results.tex` in the `tmp` directory and printed to the console.


""");