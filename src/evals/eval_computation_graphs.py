import os

from datetime import datetime
from ast import List

from src.utils import MOCK_RETURN
from src.evals.components.paper import Paper, RelatedWork, Cite, Abstract, Title, Method, Source

from symai import Symbol, Expression, Function, Interface
from symai.utils import toggle_test
from symai.components import Choice, Extract, Sequence, FileReader, RuntimeExpression, PrepareData, ExpressionBuilder
from symai.extended.seo_query_optimizer import SEOQueryOptimizer
from symai.extended.os_command import OSCommand


ACTIVE = False
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


FUNCTIONS = {}


class AppendFunction(Expression):
    def __init__(self, sequence, **kwargs):
        super().__init__(**kwargs)
        self.sequence  = sequence
        self.functions = FUNCTIONS
        self.desc_func = Function('Create a description for the custom expression.', static_context="""[Description]
Write a short and simple description in the format >>>[ ... name ... ]\n... text ...\n<<<. The description should be one or two sentence long, include a description of the expression purpose and offer one specific example how to use the expression.

[Additional Expression Information]
class Expression(Symbol):
    # initialize the expression with task specific arguments
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Here custom code
    # define the forward function with data specific arguments
    def forward(self, sym: Symbol, *args, **kwargs) -> Symbol:
        sym = self._to_symbol(sym) # ensure that sym is a Symbol
        result = None
        # Here custom code
        return result

[Example for a Paper Library Expression]
>>>[PAPER LIBRARY]
# Using the paper library index to search for papers and extract information from the paper.
expr = Interface('pinecone')
res  = expr('kernel density estimation conclusion') # example query: "kernel density estimation"
res  # str containing a list of paper citations with content
<<<

[Output Format]
>>>[ ... TODO: NAME OF THE EXPRESSION ...   ]
... # TODO: add your custom expression description here
<<<

From the output format example, you should replace all TODOs with the correct information.
""")

    def forward(self, task, *args, **kwargs):
        func = self.sequence(task, *args, **kwargs)
        desc = self.desc_func(task | func)
        expr = {desc.value: func.value}
        self.functions.update(expr)
        return self._to_symbol(expr)


FUNCTIONS.update({""">>>[SEARCH ENGINE]
# Search the web using the Google search engine to find information of topics, people, or places. Used for searching facts or information.
expr = Interface('serpapi')
res  = expr('search engine query') # example query: "weather in New York"
res # str containing the weather report
<<<""": Sequence(
    SEOQueryOptimizer(),
    Interface('serpapi')
),
""">>>[WEB BROWSER]
# Using the Selenium web driver to get the content of a web page.
expr = Interface('selenium')
res  = expr('web URL') # example URL: https://www.google.com
res  # str containing the web page source content
<<<""": Sequence(
    Extract('web URL'),
    Interface('selenium')
),
""">>>[PAPER LIBRARY]
# Using the paper library index to search for papers and extract information from papers.
expr = Interface('pinecone')
res  = expr('kernel density estimation conclusion') # example query: "kernel density estimation"
res  # str containing a list of paper citations with content
<<<""": Sequence(
    SEOQueryOptimizer(),
    Interface('pinecone', index_name='dataindex', raw_result=True)
),
""">>>[TERMINAL]
# Using the subprocess terminal module to execute shell commands.
expr = OSCommand()
expr('shell command') # example command: 'echo "text" > file.txt'
<<<""": OSCommand([
    'touch file.txt # create a new file',
    'echo "text" > file.txt # write text to file',
]),
""">>>[LARGE LANGUAGE MODEL]
# Using a large language model to build queries, provide instructions, generate content, extract patterns, and more. NOT used for searching facts or information.
expr = Function('LLM query or instruction') # example query: 'Extract the temperature value from the weather report.'
res  = expr('data') # example data: 'The temperature in New York is 20 degrees.'
res  # str containing the query result or generated text i.e. '20 degrees'
<<<""": Function('Follow the task instruction.'),
""">>>[DEFINE CUSTOM EXPRESSION]
# Define a custom sub-process using the `Expression` class and a description which can be re-used as an Expression for repetitive tasks to avoid multiple instructions for the same type of task.
<<<""": AppendFunction(Sequence(
    ExpressionBuilder(),
    RuntimeExpression(),
))})


class SequentialScheduler(Expression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a choice function to choose the correct function context.
        self.choice = Choice(FUNCTIONS.keys())
        # index of the current task
        self.index  = 1
        # pre-set optimal experiment design schedule
        self.optimal = {
            '>>> Context: Write a paper about the SymbolicAI framework from GitHub https://github.com/ExtensityAI/symbolicai. Include citations and references from the papers directory ./snippets/papers.': None,
            '>[TASK]: Create the paper and framework index from the GitHub URL and papers directory.': None,
            '>>[SUBTASK]: Use the shell to index the papers directory.': '*!./snippets/papers',
            '>[TASK]: Write a summary of the SymbolicAI framework.': None,
            '>>[SUBTASK]: Use the web browser to open the GitHub URL https://github.com/ExtensityAI/symbolicai.': None,
            '>>[SUBTASK]: Use the LLM to summarize the GitHub page.': None,
            '>[TASK]: Write the Related Work section.': None,
        }

    def forward(self, tasks, memory):
        # Generate the code for each task and subtask.
        context = tasks[0]
        task = tasks[self.index]
        # skip the helper task description
        while task.startswith('>') and '[TASK]' in task:
            print(f"Processing Task: {task}")
            self.index += 1
            task = tasks[self.index]
        # If the task is a subtask, choose the correct function context.
        if task.startswith('>>') and '[SUBTASK]' in task:
            # Choose the correct function context.
            option = self.choice(task, temperature=0.0)
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
            test   = list(self.optimal.values())[self.index]
            # Return the function and task.
            return task, test, FUNCTIONS[key], data['result']

        self.index += 1
        return None, None, None, None


class Evaluate(Expression):
    def forward(self, task, result, memory):
        # Evaluate the result of the program.
        sim     = task.similarity(result)
        success = sim > 0.5 and 'finished successfully' in memory.join()
        memory.append(f"EVAL: {task} | Similarity: {sim} | Success: {success}")
        # TODO: ...


class Agent(Expression):
    def __init__(self, halt_threshold: float = 0.85,
                 max_iterations: int = 1,
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
        task        = 'start'
        # Execute the program
        while task is not None:
            # Schedule the next task.
            task, test, func, data = self.schedule(self.tasks, self.memory)
            # Execute sub-routine until the target goal is reached.
            while task is not None and sim < self.halt_threshold and n_iter < self.max_iterations:
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


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_program():
    expr   = Agent()
    reader = FileReader()
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    target = reader(os.path.join(cur_file_dir, 'snippets/richard_feynman_summary.txt'))
    res    = expr("Write a paper about the SymbolicAI framework from GitHub https://github.com/ExtensityAI/symbolicai. Include citations and references from the papers directory ./snippets/papers.")
    print(res)
    return True, {'scores': [1.0]}


SUB_ROUTINE_ACTIVE = False


@toggle_test(SUB_ROUTINE_ACTIVE, default=MOCK_RETURN)
def test_sub_routine_custom_expression():
    # define the task
    task   = "Create a query Expression that is initializes a Function with a prompt and processes a data Symbol based on the custom Function."
    # choose the correct function context
    choice = Choice(FUNCTIONS.keys())
    option = choice(task, temperature=0.0)
    option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
    key    = list(FUNCTIONS.keys())[option]
    func   = FUNCTIONS[key]
    # run the sub-routine function
    entry  = func(task)
    expr   = list(entry.values())[0]('extract the temperature value from the weather report.')
    val    = expr('The temperature in New York is 20 degrees.')
    # check the sub-routine result
    res    = '20' in val
    return res, {'scores': [float(res)]}


@toggle_test(SUB_ROUTINE_ACTIVE, default=MOCK_RETURN)
def test_sub_routine_search_engine():
    # define the task
    task   = "How is the weather in New York today?"
    # choose the correct function context
    choice = Choice(FUNCTIONS.keys())
    option = choice(task, temperature=0.0)
    option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
    key    = list(FUNCTIONS.keys())[option]
    func   = FUNCTIONS[key]
    # run the sub-routine function
    res    = func(task)
    # check the sub-routine result
    res    = 'Success' == res.raw['search_metadata']['status']
    return res, {'scores': [float(res)]}


@toggle_test(SUB_ROUTINE_ACTIVE, default=MOCK_RETURN)
def test_sub_routine_web_crawler():
    # define the task
    task   = "Open up the website https://www.cnbc.com/investing/"
    # choose the correct function context
    choice = Choice(FUNCTIONS.keys())
    option = choice(task, temperature=0.0)
    option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
    key    = list(FUNCTIONS.keys())[option]
    func   = FUNCTIONS[key]
    # run the sub-routine function
    res    = func(task)
    # check the sub-routine result
    res    = 'CNBC' in res
    return res, {'scores': [float(res)]}


@toggle_test(True, default=MOCK_RETURN)
def test_sub_routine_paper_indexer():
    # define the task
    task   = "Explain the central concepts in programming language theory used in SymbolicAI using the indexed papers."
    # choose the correct function context
    choice = Choice(FUNCTIONS.keys())
    option = choice(task, temperature=0.0)
    option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
    key    = list(FUNCTIONS.keys())[option]
    func   = FUNCTIONS[key]
    # run the sub-routine function
    res    = func(task)
    # check the sub-routine result
    res    = 'Chomsky' in res.join()
    return res, {'scores': [float(res)]}


@toggle_test(SUB_ROUTINE_ACTIVE, default=MOCK_RETURN)
def test_sub_routine_os_commands():
    # define the task
    task   = "Create a new text file named `results/test.txt` in the `results` directory and write the text `Hello World!` to the file."
    # choose the correct function context
    choice = Choice(FUNCTIONS.keys())
    option = choice(task, temperature=0.0)
    option = Symbol(option).similarity(Symbol.symbols(*FUNCTIONS.keys())).argmax()
    key    = list(FUNCTIONS.keys())[option]
    func   = FUNCTIONS[key]
    os.makedirs('results', exist_ok=True)
    # run the sub-routine function
    func(task)
    # check the sub-routine result
    res    = os.path.exists('results/test.txt')
    # clean up
    os.remove('results/test.txt')
    return res, {'scores': [float(res)]}


@toggle_test(True, default=MOCK_RETURN)
def test_sub_routine_create_paper():
    # define the task
    task   = "Write a paper about the SymbolicAI framework from GitHub https://github.com/ExtensityAI/symbolicai. Include citations and references from the papers directory ./snippets/papers."
    # choose the correct function context
    expr   = Paper(
        Method(
            Source(file_link=['src/evals/snippets/assets/symbolicai_docs.txt']),
        ),
        RelatedWork(
            Cite(file_link='src/evals/snippets/bib/related_work/laird87.txt'),
            Cite(file_link='src/evals/snippets/bib/related_work/mccarthy06.txt'),
            Cite(file_link='src/evals/snippets/bib/related_work/newell56.txt'),
            Cite(file_link='src/evals/snippets/bib/related_work/newell57.txt'),
            Cite(file_link='src/evals/snippets/bib/related_work/newell72.txt'),
        ),
        Abstract(),
        Title(),
    )
    res    = expr(task)

    return res, {'scores': [float(res)]}