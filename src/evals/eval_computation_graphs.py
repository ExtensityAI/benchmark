from symai import Symbol, Expression, Function, Interface
from symai.components import Choice, Execute, Sequence


DataDictionary = {}
TaskList       = []


class Store(Expression):
    def forward(self, key, data, *args, **kwargs):
        DataDictionary[key] = data


class Recall(Expression):
    def forward(self, key, *args, **kwargs):
        return DataDictionary.get(key, None)


class TaskExtraction(Function):
    @property
    def static_context(self):
        return """[Description]
Your goal is to extract the tasks and subtasks from the system instruction as a hierarchical structure.
Each task and subtask should be represented as in a functional block as shown in the example below.
Try to define the task and subtask description in one sentence.

[Example]
Instruction: "Write a document about the history of AI."

>1.[TASK]: Create a new text document.
>>1.1.[SUBTASK]: Use the LLM to generate shell command that creates a new text document named `history_of_ai.txt` in the Documents folder of the user home directory.
>>1.2.[SUBTASK]: Execute the command in the terminal.
>2.[TASK]: Google the history of AI.
>>2.1.[SUBTASK]: Create a search engine query using the LLM.
>>2.2.[SUBTASK]: Run the query in the search engine.
>3.[TASK]: Generate the text.
>>3.1.[SUBTASK]: Use the LLM to summarize the search engine results and write the tile and introduction to the document.
>>3.2.[SUBTASK]: Use the LLM to generate the rest of the document.
>>3.3.[SUBTASK]: Use the LLM to generate a shell command that saves the document text to the file.
>>3.4.[SUBTASK]: Execute the command in the terminal.
"""


class CodeGeneration(Function):
    @property
    def static_context(self):
        return """[Description]
Your goal is to generate the code of the forward function following the instruction of the extracted task and task description. Expect that all imports are already defined. Only produce the code for the TODO section as shown below:

[Template]
class TemplateExpression(Expression):
    def forward(self, instruct, *args, **kwargs):
        result = None
        ```python
        # TODO: Place here the generated code.
        ```
        return result

All code is executed in the same Python process. Expect all task expressions and functions to be defined in the same process. Generate the code within a ```python # TODO: ``` code block as shown in the example.
"""


FUNCTIONS = {"""
>>>[SEARCH ENGINE]
# Search the web using the Google search engine.
expr = Interface('serpapi')
res  = expr('search engine query') # example query: "weather in New York"
res # str containing the weather report
<<<""": Interface('serpapi'),
"""
>>>[WEB BROWSER]
# Using the Selenium web driver to get the web page source content.
expr = Interface('selenium')
res  = expr('web URL') # example URL: https://www.google.com
res  # str containing the web page source content
<<<""": Interface('selenium'),
"""
>>>[TERMINAL]
# Using the subprocess terminal module to execute shell commands.
expr = Interface('terminal')
expr('shell command') # example command: 'ls -l'
<<<""": Interface('terminal'),
"""
>>>[LARGE LANGUAGE MODEL]
# Using a large language model to query information, follow instructions, or generate text.
expr = Function('LLM query or instruction') # example query: 'Extract temperature from the weather report.'
res  = expr()
res  # str containing the query result or generated text
<<<""": Function('Follow the user instructions:'),
"""
>>>[TASK EXTRACTION]
# Using a large language model to extract tasks from the system instruction.
expr = TaskExtraction()
res  = expr('system instruction') # example instruction: 'Get the weather in New York.'
res  # str containing the extracted tasks and subtasks
<<<""": TaskExtraction('Extract all tasks from the user query:'),
"""
>>>[DEFINE CUSTOM EXPRESSION]
# Define a custom expression using the `Expression` class and a static_context description which can be re-used as an Expression.
class MyExpression(Expression): # class name of the custom expression
    def static_context(self):
        return '''Multi-line description of the expression specifying the context of the expression with examples, instructions, desired return format and other information.'''
<<<""": Sequence(
    CodeGeneration('Define a custom expression:'),
    Execute(enclosure=True)
)}


# TODO:
META_INSTRUCTIONS = {"""
>>>[STORE DATA]
# Using the `store` function to store data in memory.
expr = Store()
expr('key', 'value') # example key: 'weather report', value: 'The weather in New York is 70 degrees.'
<<<""": Store(),
"""
>>>[RECALL DATA]
# Using the `recall` function to recall data from memory.
expr = Recall()
res  = expr('key') # example key: 'weather report'
res  # str containing the recalled data
<<<""": Recall()
}


class Program(Expression):
    def __init__(self):
        super().__init__()
        # Function to extract all sub-tasks from the system instruction.
        self.task_extraction = TaskExtraction('Extract all tasks from the user query:')
        # Create a choice function to choose the correct function context.
        self.choice          = Choice(FUNCTIONS.keys())
        # Execute instruction.
        self.runner          = Execute(enclosure=True)

    def forward(self, instruct, *args, **kwargs):
        global TaskList
        # Extract all tasks from the system instruction.
        tasks    = self.task_extraction(instruct)
        TaskList = tasks.split('\n')
        print(f"Extracted Tasks: {TaskList}")
        value    = None
        # Generate the code for each task and subtask.
        for task in TaskList: # TODO: make a model do the scheduling
            if task.startswith('>') and '[TASK]' in str(task):
                print(f"Processing Task: {task}")
            elif task.startswith('>>') and '[SUBTASK]' in str(task):
                option = self.choice(task)
                option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
                # Run the expression
                key    = list(FUNCTIONS.keys())[option]
                value  = FUNCTIONS[key](task)
        # Return the extracted tasks.
        return value


class MetaProgram(Expression):
    pass # TODO: implement the meta-programming interface


def test_program():
    expr = Program()
    res  = expr('Play Taylor Swift on Spotify.')
    print(res)
    return True, {'scores': [1.0]}


# def test_meta_program():
#     expr = MetaProgram()
#     res  = expr('Play Taylor Swift on Spotify.')
#     print(res)
#     return True, {'scores': [1.0]}
