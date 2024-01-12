import os

from datetime import datetime
from ast import List

from symai import Symbol, Expression, Function, Interface
from symai.components import Choice, Execute, Sequence, FileReader
from symai.processor import ProcessorPipeline
from symai.constraints import DictFormatConstraint
from symai.post_processors import StripPostProcessor, CodeExtractPostProcessor, JsonTruncateMarkdownPostProcessor
from symai.pre_processors import PreProcessor


DataDictionary = Symbol({})


class Store(Expression):
    def forward(self, key, data, *args, **kwargs):
        # currently use hard-matching
        DataDictionary.value[key] = data


class Recall(Expression):
    def forward(self, key, *args, **kwargs):
        # currently use hard-matching
        try:
            return DataDictionary[key]
        except KeyError:
            return None


class Memory(Function):
    def __init__(self, **kwargs):
        super().__init__("Your goal is to create a short description of the program using the key and execution flow of the context for the stack trace:", **kwargs)
        self._store             = Store()
        self._recall            = Recall(iterate_nesy=True) # allow to iterate with dict[key] over the neuro-symbolic engine
        self._value: List[str]  = []

    def forward(self, *args, **kwargs):
        # use store and recall functions to store and recall data from memory
        raise NotImplementedError

    def store(self, key, data):
        result = super().forward(key | data)
        self._store(key, data)
        # get execution timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.value.append(f"EXEC#: {len(self.value)+1} TIMESTAMP:{timestamp} | Store data: {key} = {result}")

    def recall(self, key):
        result = super().forward(key)
        self._recall(key)
        self.value.append(f"Recall data: {key} = {result}")
        return result

    @property
    def static_context(self):
        return """[Description]
Your goal is to store and recall data from memory using the `store` and `recall` functions.
The `store` function takes in a `key` and `data` argument and stores the data in a dictionary.
The `recall` function takes in a `key` argument and returns the data stored in the dictionary.
The `trace` function keeps track of the `key` and from which context the `store` and `recall` functions were called to function like a stack trace and description of the execution flow.
"""


class TaskExtraction(Function):
    @property
    def static_context(self):
        return """[Description]
Your goal is to extract the tasks and subtasks from the system instruction as a hierarchical structure.
Each task and subtask should be represented as in a functional block as shown in the example below.
Try to define the task and subtask description in one sentence.

[Capabilities]
%s

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
""" % '\n'.join([k for k in FUNCTIONS.keys()])


class CodeGeneration(Function):
    def __init__(self, **kwargs):
        super().__init__('Generate the code following the instructions:', **kwargs)
        self.processors = ProcessorPipeline([StripPostProcessor(), CodeExtractPostProcessor()])

    def forward(self, instruct, *args, **kwargs):
        result = super().forward(instruct)
        return self.processors(str(result), None)

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

All code is executed in the same Python process. Expect all task expressions and functions to be defined in the same process. Generate the code within a ```python # TODO: ``` code block as shown in the example. The code must be self-contained, include all imports and executable.
"""


FUNCTIONS = {"""
>>>[SEARCH ENGINE]
# Search the web using the Google search engine to find information of topics, people, or places. Used for searching facts or information.
expr = Interface('serpapi')
res  = expr('search engine query') # example query: "weather in New York"
res # str containing the weather report
<<<""": Interface('serpapi'),
"""
>>>[WEB BROWSER]
# Using the Selenium web driver to get the content of a web page.
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
# Using a large language model to build queries, provide instructions, generate content, extract patterns, and more. NOT used for searching facts or information.
expr = Function('LLM query or instruction') # example query: 'Extract the temperature value from the weather report.'
res  = expr('data') # example data: 'The temperature in New York is 20 degrees.'
res  # str containing the query result or generated text i.e. '20 degrees'
<<<""": Function('Follow the instruction of the task.'),
"""
>>>[DEFINE CUSTOM EXPRESSION]
# Define a custom sub-process and code using the `Expression` class and a static_context description which can be re-used as an Expression for repetitive tasks to avoid multiple instructions for the same type of task.
class MyExpression(Expression): # class name of the custom expression
    def static_context(self):
        return '''Multi-line description of the expression specifying the context of the expression with examples, instructions, desired return format and other information.'''
<<<""": CodeGeneration()}


class PrepareData(Function):
    class PrepareDataPreProcessor(PreProcessor):
        def __call__(self, argument):
            assert argument.prop.context is not None
            instruct = argument.prop.prompt
            context  = argument.prop.context
            return """{
    'context': '%s',
    'instruction': '%s',
    'result': 'TODO: Replace this with the expected result.'
}""" % (context, instruct)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_processors  = [self.PrepareDataPreProcessor()]
        self.constraints     = [DictFormatConstraint({ 'result': '<the data>' })]
        self.post_processors = [JsonTruncateMarkdownPostProcessor()]
        self.return_type     = dict # constraint to cast the result to a dict

    @property
    def static_context(self):
        return """[CONTEXT]
Your goal is to prepare the data for the next task instruction. The data should follow the format of the task description based on the given context. Replace the `TODO` section with the expected result of the data preparation. Only provide the 'result' json-key as follows: ```json\n{ 'result': 'TODO:...' }\n```

[GENERAL TEMPLATE]
```json
{
    'context': 'The general context of the task.',
    'instruction': 'The next instruction of the task for the data preparation.',
    'result': 'The expected result of the data preparation.'
}
```

[EXAMPLE]
[Instruction]:
{
    'context': 'Write a document about the history of AI and include references to the following people: Alan Turing, John McCarthy, Marvin Minsky, and Yoshua Bengio.',
    'instruction': 'Google the history of AI for Alan Turing',
    'result': 'TODO'
}

[Result]:
```json
{
    'result': 'Alan Turing history of AI'
}
```
"""


class SequentialScheduler(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a choice function to choose the correct function context.
        self.choice = Choice(FUNCTIONS.keys())
        # index of the current task
        self.index  = 1

    def forward(self, tasks, memory):
        # Generate the code for each task and subtask.
        context = tasks[0]
        task = tasks[self.index] # TODO: make a model do the scheduling
        while task.startswith('>') and '[TASK]' in task:
            print(f"Processing Task: {task}")
            self.index += 1
            task = tasks[self.index]
        # If the task is a subtask, choose the correct function context.
        if task.startswith('>>') and '[SUBTASK]' in task:
            # Choose the correct function context.
            option = self.choice(task)
            option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
            # Run the expression
            key    = list(FUNCTIONS.keys())[option]
            # Use the memory to store the result of the expression.
            memory.store(task, key)
            # Prepare the query for the next task.
            func   = PrepareData(prompt=task, context=context)
            data   = func(context) # concat the context with the task
            # increment the task index
            self.index += 1
            # Return the function and task.
            return task, FUNCTIONS[key], data['result']

        self.index += 1
        return None, None, None


class Evaluate(Expression):
    def forward(self, task, result, memory):
        # Evaluate the result of the program.
        sim     = task.similarity(result)
        success = sim > 0.5 and 'finished successfully' in memory.join()
        memory.append(f"EVAL: {task} | Similarity: {sim} | Success: {success}")
        # TODO: ...


class Program(Expression):
    def __init__(self, halt_threshold: float = 0.85,
                 max_iterations: int = 3,
                 scheduler = SequentialScheduler,
                 **kwargs):
        super().__init__(**kwargs)
        # halt threshold
        self.halt_threshold  = halt_threshold
        # max iterations
        self.max_iterations  = max_iterations
        # buffer memory
        self.memory          = Memory(iterate_nesy=True)
        # scheduler
        self.schedule        = scheduler()
        # evaluation
        self.eval            = Evaluate()
        # Function to extract all sub-tasks from the system instruction.
        self.task_extraction = TaskExtraction('Extract all tasks from the user query:')
        # tasks and subtasks
        self.tasks           = None
        # target
        self.target          = None

    def extract(self, instruct):
        # Extract all tasks from the system instruction.
        tasks     = self.task_extraction(instruct)
        tasks     = tasks.split('\n')
        tasks     = [t for t in tasks if t.strip() != '']
        task_list = ['>>> Context:' | Symbol(instruct), *tasks]
        print(f"Extracted Tasks: {task_list}")
        return task_list

    def forward(self, instruct):
        # Set the target goal.
        self.target = Symbol(instruct)
        # Extract all tasks from the system instruction.
        self.tasks  = self.extract(instruct)
        sim         = 0.0
        n_iter      = 0
        result      = None

        # Execute the program until the target goal is reached.
        while sim < self.halt_threshold and n_iter < self.max_iterations:
            # Schedule the next task.
            task, func, data = self.schedule(self.tasks, self.memory)
            # Execute the task.
            try:
                result = func(data)
                self.memory.append(f"EXEC SUCCESS: {task}")
            except:
                result = None
                self.memory.append(f"ERROR: {func.__class__} raised an exception. {task}")
            # Evaluate the result of the program.
            self.eval(task, result, self.memory)
            # update the similarity
            sim        = Symbol(result).similarity(self.target)
            # increment the iteration counter
            n_iter    += 1

        # Return the final result.
        return result


def test_program():
    expr   = Program()
    reader = FileReader()
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    target = reader(os.path.join(cur_file_dir, 'snippets/richard_feynman_summary.txt'))
    res    = expr("Write an article about Richard Feynman and who his doctoral students were in Markdown format.")
    print(res)
    return True, {'scores': [1.0]}


if __name__ == '__main__':
    prompt  = 'Create a search query for the weather in New York.'
    context = 'Report on the weather to compare between multiple cities including New York, London, and Paris.'
    func    = PrepareData(prompt, context=context)
    res     = func(context)
    print(res)
