"""
Some details…

We want to evaluate a workflow based on a non-linear execution of tasks, i.e. the tasks are not executed in order, or new tasks can be dynamically inserted into the workflow and jumped to.
Because we are still in the infancy of even constructing a linear scheduler with a LLM planing system, we will only consider a linear workflow for now.

So, we start with a high-level workflow description, which consists of a list of tasks. We call this a plan.
To perform the experiment, we use an expected plan. We also have a set of plans similar to the expected plan, which are trajectories in the solution space, as well as the plan that the LLM predicts for a specific seed.
We score the predicted plan against the expected plan, and the trajectories, then we continue to the next phase in which we use the expected plan to execute the tasks.

At each step, the LLM will get the goal of the workflow, the tasks, the current progress, and a query asking for the next task to execute.
If the LLM is not able to predict the next task, it will return a failure, and the expected plan will be used to execute the next task.
We do this until the list of tasks in exhausted, scoring the LLM at each step.
"""

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from symai import Function, Interface, Symbol
from symai.backend.engines.index.engine_vectordb import VectorDBIndexEngine
from symai.components import FileReader
from symai.functional import EngineRepository
from symai.memory import SlidingWindowStringConcatMemory
from symai.utils import toggle_test

from src.utils import MOCK_RETURN, RANDOMNESS, normalize

ACTIVE = True
LOG = lambda msg, active=not ACTIVE: print(f"DEBUG:root:\n{msg}") if active else ""

reader = FileReader()
dir_path = Path(__file__).parent.absolute() / "snippets"
GOOGLE_RESULTS = reader((dir_path / "google_organic_results_20240121_query=Search-for-U-235.txt").as_posix())
WIKI_PAGE      = reader((dir_path / "wiki_page_20240121.txt").as_posix())

GOAL = "Search for U-235, access the Wikipedia page, find out what is the half-life of U-235, and then take the binary logarithm of the half-life."

CAPABILITIES = {
    "Google Search: to be used for searching facts or information." : Interface("serpapi"),
    "Browser: to be used for opening web pages." : Interface("selenium"),
    "Wolfram Alpha: to be used for getting precise answers to numerical calculations." : Interface("wolframalpha"),
    "Large Language Model (LLM): to be used for extracting information." : None
}

PLAN_GENERATION_TEMPLATE = """
Given a goal, write a plan that you would follow to achieve it.

Goal:
Develop a machine learning model to predict stock prices using data from Yahoo Finance. The model should be able to forecast prices for the next five days based on historical data.

Plan:
>Task 1: Gather and preprocess historical stock data
>>Subtask 1.1: Use Python to scrape historical stock data from Yahoo Finance
>>Subtask 1.2: Clean and preprocess the data for model training (handling missing values, normalizing data, etc.)
>>Subtask 1.3: Split the data into training and validation sets
>Task 2: Develop the prediction model
>>Subtask 2.1: Select and implement a suitable machine learning algorithm (e.g., LSTM, ARIMA)
>>Subtask 2.2: Train the model on the preprocessed training set
>Task 3: Validate the model
>>Subtask 3.1: Evaluate the model's performance using metrics like RMSE and MAE
>Task 4: Forecast future stock prices
>>Subtask 4.1: Use the model to predict stock prices for the next five days
"""

EXPECTED_PLAN = deque([
    {">Task 1: Search for U-235" : [
        ">>Subtask 1.1: Extract the Wikipedia URL"
    ]},
    {">Task 2: Access the Wikipedia URL" : [
        ">>Subtask 2.1: Extract the half-life of U-235 from the page",
        ">>Subtask 2.2: Extract the number"
    ]},
    {">Task 3: Take the binary logarithm" : [
        ">>Subtask 3.1: Extract the number"
    ]}
])

TASK_EXTRACTION_TEMPLATE = f"""
This is your goal:
Write a paper about the SymbolicAI framework from GitHub https://github.com/ExtensityAI/symbolicai. Include citations and references from the papers directory `./snippets/papers`.

History of tasks that have been executed:
[EXECUTED SUCCESSFULLY@21/01/2024 15:38:45:867949]:
<<<
>Task 1: Create the paper and framework index from the GitHub URL and papers directory
>>>
[EXECUTED SUCCESSFULLY@21/01/2024 15:38:50:194266]:
<<<
>>Subtask 1.1: Use the shell to index the papers directory
>>>
[EXECUTED SUCCESSFULLY@21/01/2024 15:38:55:194266]:
<<<
>>Subtask 1.2: Use the shell to index the GitHub URL
>>>

This is the pool of tasks that are left to be executed:
>Task 2: Write a summary of the SymbolicAI framework
>>Subtask 2.1: Use the web browser to open the GitHub URL https://github.com/ExtensityAI/symbolicai
>>Subtask 2.2: Summarize the GitHub page
>Task 3: Write the Related Work section

Answer to the query "What is the next task to execute?"
Based on the pool of tasks, the correct answer is:
>Task 2: Write a summary of the SymbolicAI framework
"""

SOLUTION = {
    ">Task 1: Search for U-235": {
        "expected_interface": Interface("serpapi").__class__.__name__,
        "expected_result": Symbol(GOOGLE_RESULTS),
        "expected_next_task": ">>Subtask 1.1: Extract the Wikipedia URL"
    },
    ">>Subtask 1.1: Extract the Wikipedia URL": {
        "expected_interface": "NoneType",
        "expected_result": Symbol("https://en.wikipedia.org/wiki/Uranium-235"),
        "expected_next_task": ">Task 2: Access the Wikipedia URL"
    },
    ">Task 2: Access the Wikipedia URL" : {
        "expected_interface": Interface("selenium").__class__.__name__,
        "expected_result": Symbol(WIKI_PAGE),
        "expected_next_task": ">>Subtask 2.1: Extract the half-life of U-235 from the page"
    },
    ">>Subtask 2.1: Extract the half-life of U-235 from the page": {
        "expected_interface": "NoneType",
        "expected_result": Symbol("703.8 million years"),
        "expected_next_task": ">>Subtask 2.2: Extract the number"
    },
    ">>Subtask 2.2: Extract the number" : {
        "expected_interface": "NoneType",
        "expected_result": Symbol("703.8"),
        "expected_next_task": ">Task 3: Take the binary logarithm"
    },
    ">Task 3: Take the binary logarithm" : {
        "expected_interface": Interface("wolframalpha").__class__.__name__,
        "expected_result": Symbol("9.45902 binomial(n, r)"),
        "expected_next_task": ">>Subtask 3.1: Extract the number"
    },
    ">>Subtask 3.1: Extract the number" : {
        "expected_interface": "NoneType",
        "expected_result": Symbol("9.45902"),
        "expected_next_task": None
    },
    GOAL: {
    "trajectories": [
        """
>Task 1: Use the search engine and lookup U-235
>>Subtask 1.1: Retrieve the Wikipedia link from the results
>Task 2: Access the Wikipedia link and extract the half-life of U-235
>>Subtask 2.1: Retrieve the half-life from the wikipedia article
>>Subtask 2.2: Retrieve the exact number from the result
>Task 3: Take the binary logarithm of the half-life of U-235
>>Subtask 3.1: Retrieve the exact number from the result
""",
        """
>Task 1: Search for U-235
>>Subtask 1.1: Extract the Wikipedia URL
>Task 2: Open the Wikipedia URL
>>Subtask 2.1: Extract the half-life
>>Subtask 2.2: Extract the number
>Task 3: Compute the binary logarithm
>>Subtask 3.1: Extract the number
"""
        ]
    }
}


@dataclass
class Setup:
    goal: str
    expected_plan: deque
    capabilities: Dict
    task_extraction_template: str
    plan_generation_template: str
    solution: Dict


class ToolKit:
    def __init__(self, capabilities: Dict):
        self.capabilities = capabilities

    def pick_tool(self, query: str) -> Optional[Interface]:
        "Pick the best tool for the task at hand."

        # LLM has to reflect using first person, then based on the reflection, it must choose the correct interface.
        f = Function(f"Reflect and narrate in first person which of the following capabilities would be best for the task at hand: {self.capabilities.keys()}")
        # We summarize the reflection to increase the chance of the LLM picking the correct interface.
        reflection = f(query).summarize()
        tool = reflection.similarity(Symbol.symbols(*self.capabilities.keys())).argmax()
        interface = self.capabilities[list(self.capabilities.keys())[tool]]
        return interface if interface.__class__.__name__ != "NoneType" else None

    def apply_tool(self, tool: Interface, query: Symbol, payload: Symbol = None) -> Symbol:
        "Apply the tool to the query."

        if tool.__class__.__name__ == "serpapi":
            result = tool(query)
            return Symbol(result.raw.organic_results.to_list())

        if tool.__class__.__name__ == "selenium":
            result = tool(payload)
            return result

        if tool.__class__.__name__ == "wolframalpha":
            result = tool(query << payload)
            return result

        return


class TaskExtractor:
    def __init__(self, task_extraction_template: str, choices: List):
        self.task_extraction_template = task_extraction_template
        self.choices  = choices

    def pick_next_task(self, data: str) -> Optional[str]:
        f = Function(f"Reflect and narrate in first person which of the following tasks would be best to execute next: {data}", examples=self.task_extraction_template)
        reflection = f(data)
        task = reflection.similarity(Symbol.symbols(*self.choices)).argmax()
        return self.choices[task]


class PlanGenerator:
    def __init__(self, plan_generation_template: str):
        self.plan_generation_template = plan_generation_template

    def generate_plan(self, goal: str) -> str:
        template = f"""
        Given a goal, write a plan that you would follow to achieve it.

        Goal:
        {goal}

        Plan:
        """
        f = Function(template, examples=self.plan_generation_template)
        plan = f()
        return plan


class Memory(SlidingWindowStringConcatMemory):
    def __init__(self, token_ratio: float = 0.9, use_long_term_mem: bool = False):
        super().__init__(token_ratio=token_ratio)

        if use_long_term_mem:
            self._register_local_index()
            self.long_term_mem = Interface("vectordb")
        else:
            self.long_term_mem = None

    def store(self, query: str, where: str):
        if isinstance(query, str):
            query = Symbol(query)

        if where == "long_term_mem":
            return self.long_term_mem(query, operation="add")

        if where == "short_term_mem":
            return super().store(query)

    def recall(self, query: str, where: str):
        if isinstance(query, str):
            query = Symbol(query)

        if where == "long_term_mem":
            return self.long_term_mem(query, operation="search")

        if where == "short_term_mem":
            return super().recall(query)

    def _register_local_index(self):
        EngineRepository.register_from_plugin('embedding', plugin='ExtensityAI/embeddings', kwargs={'model': 'all-mpnet-base-v2'}, allow_engine_override=True)
        EngineRepository.register('index', VectorDBIndexEngine(index_name='dataindex', index_dims=768, index_top_k=5))


class Evaluator:
    def __init__(self, goal: str, solution: Dict, results: Dict):
        self.goal     = goal
        self.solution = solution
        self.results  = results

        # Compute the score for each task.
        self.scoring = defaultdict(list)

    def collect(self):
        for task in self.solution:
            res = self.results[task]
            sol = self.solution[task]
            expected_result = Symbol(sol["expected_result"])
            if task == self.goal:
                # Only for the goal we have trajectories.
                trajectories = Symbol(sol["trajectories"]).mean()
                rand_score   = expected_result.measure(Symbol(RANDOMNESS).mean())
                base_score   = expected_result.measure(trajectories)
                plan_score   = expected_result.measure(res["predicted_result"], normalize=normalize(base_score, rand_score)).value
                self.scoring[task].append(plan_score)
                self.scoring[task] += res["execution"]
            else:
                expected_interface = Symbol(sol["expected_interface"])
                expected_next_task = Symbol(sol["expected_next_task"])
                if res.get("predicted_result") is not None:
                    # We have a predicted result from the LLM.
                    predicted_result_score = expected_result.measure(res["predicted_result"]).value
                    self.scoring[task].append(predicted_result_score)

                predicted_interface_score = expected_interface.measure(res["predicted_interface"]).value
                predicted_next_task_score = expected_next_task.measure(res["predicted_next_task"]).value
                self.scoring[task].append(predicted_interface_score)
                self.scoring[task].append(predicted_next_task_score)
                self.scoring[task] += res["execution"]

        return self.scoring


class SequentialScheduler:
    def __init__(self, setup: Setup):
        self.goal           = setup.goal
        self.expected_plan  = setup.expected_plan
        self.solution       = setup.solution
        self.memory         = Memory()
        self.toolkit        = ToolKit(setup.capabilities)
        self.task_extractor = TaskExtractor(setup.task_extraction_template, choices=self._plan_as_list())
        self.plan_generator = PlanGenerator(setup.plan_generation_template)
        self.setup          = setup
        self.results        = defaultdict(lambda: defaultdict(list))

        # We generate a plan based on the goal to see how the LLM performs against the expected plan.
        self.generate_plan()

        # We store in one variable the results of the last task to propagate to the next task.
        self._payload = None
        self._pool    = deque(self._plan_as_list())

    def unfold(self):
        while self.expected_plan:
            # We start with the first task in the plan.
            task = self.expected_plan.popleft()
            # We get the task name.
            task_name = list(task.keys())[0]
            # We get the subtasks if any.
            subtasks = task.get(task_name)
            # We execute the task.
            LOG(self._build_tag("TASK", task_name))
            LOG(self._build_tag("SUBTASKS", subtasks))
            self._execute(task_name, subtasks)

            # We continue unfolding the plan…
            self.unfold()

        return self.results

    def generate_plan(self):
        try:
            predicted_plan = self.plan_generator.generate_plan(self.goal)
            self.results[self.goal]["predicted_result"] = predicted_plan.value
            self.results[self.goal]["execution"].append(1)
        except Exception as e:
            self.results[self.goal]["predicted_result"] = str(e)
            self.results[self.goal]["execution"].append(0)

        # We store the expected plan.
        self.solution[self.goal]["expected_result"] = self._plan_as_str()

    def _execute(self, task_name: str, subtasks: Optional[List] = None):
        # We get the tool to execute the task.
        try:
            # assert False, "This is a test."
            tool = self.toolkit.pick_tool(task_name)
            self.results[task_name]["predicted_interface"] = tool.__class__.__name__ #@NOTE: Here we score whether the LLM picked the correct interface.
            self.results[task_name]["execution"].append(1)
        except Exception as e:
            tool = self.solution[task_name]["expected_interface"]
            tool = None if tool == "NoneType" else tool
            self.results[task_name]["predicted_interface"] = str(e) #@NOTE: Bad luck is still no luck at all.
            self.results[task_name]["execution"].append(0)

        query = Symbol(task_name.split(":")[1].strip())

        #@NOTE: Here we score whether the LLM picked the correct result and use the ideal result to continue the execution.
        #       Moreover, we don't score tools. They must work, otherwise why are they tools?
        if tool is None:
            LOG(f"LLM is used to execute the query <{query}>.")
            try:
                # assert False, "This is a test."
                result = self._payload.query(query, temperature=0.0)
                self.results[task_name]["predicted_result"] = str(result.value) #@NOTE: Here we score whether the LLM picked the correct interface.
                self.results[task_name]["execution"].append(1)
            except Exception as e:
                result = self.solution[task_name]["expected_result"]
                self.results[task_name]["predicted_result"] = str(e) #@NOTE: Bad luck is still no luck at all.
                self.results[task_name]["execution"].append(0)

            self._payload = self.solution[task_name]["expected_result"]
            self._update_memory_buffer(self._payload, task_name)
        else:
            LOG(f"{tool.__class__.__name__} is used to execute the query <{query}>.")
            # result = self.toolkit.apply_tool(tool, query, payload=self._payload) #@NOTE: We don't need to execute this because we already have the expected result. Kept for consistency.
            self._payload = self.solution[task_name]["expected_result"]
            self._update_memory_buffer(self._payload, task_name)

        # We remove the task from the pool.
        self._pool.remove(task_name)

        # We get the next task to execute.
        try:
            # assert False, "This is a test."
            next_task = self.task_extractor.pick_next_task(self._prepare_data())
            self.results[task_name]["predicted_next_task"] = next_task #@NOTE: Here we score whether the LLM picked the correct next task.
            self.results[task_name]["execution"].append(1)
        except Exception as e:
            next_task = self.solution[task_name]["expected_next_task"]
            self.results[task_name]["predicted_next_task"] = str(e) #@NOTE: Bad luck is still no luck at all.
            self.results[task_name]["execution"].append(0)

        LOG(self._build_tag("NEXT TASK", next_task))

        if subtasks is not None:
            # We execute the subtasks.
            for subtask in subtasks:
                self._execute(subtask)

        # We store the fact that we executed the task.
        if self.memory.long_term_mem is not None:
            self.memory.store(task_name, where="long_term_mem")

    def _update_memory_buffer(self, result: str, task_name: str):
        # We store the result in the short-term memory.
        result = self._build_tag(f"EXECUTED SUCCESSFULLY", task_name)
        self.memory.store(result, where="short_term_mem")
        LOG(self._build_tag("MEMORY BUFFER", "".join(self.memory.history())))

    def _prepare_data(self):
        # We get the history.
        history = "".join(self.memory.history())
        pool    = "\n".join(self._pool)
        template = f"""
This is your goal.
{self.goal}

History of tasks that have been executed.
{history}
This is the pool of tasks that are left to be executed.
{pool if pool else "The pool is empty, there are no tasks left."}

Answer to the query "What is the next task to execute?"
Based on the pool of tasks, the correct answer is:
"""
        LOG(self._build_tag("TEMPLATE", template))
        return template

    def _build_tag(self, tag: str, query: str) -> str:
        # get timestamp in string format
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        return str(f"[{tag}@{timestamp}]:\n<<<\n{str(query)}\n>>>\n")

    def _plan_as_str(self):
        s = ""
        for task in self.expected_plan:
           for k in task:
               subtask = task.get(k)
               s += k
               s += "\n"
               if subtask is not None:
                   for st in subtask:
                       s += st
                       s += "\n"
        return s

    def _plan_as_list(self):
        return list(filter(lambda x: x, self._plan_as_str().split("\n"))) # Filter empty strings


@toggle_test(ACTIVE, default=MOCK_RETURN)
def test_workflow(aggregate):
    setup = Setup(
        goal                     = GOAL,
        expected_plan            = EXPECTED_PLAN,
        capabilities             = CAPABILITIES,
        task_extraction_template = TASK_EXTRACTION_TEMPLATE,
        plan_generation_template = PLAN_GENERATION_TEMPLATE,
        solution                 = SOLUTION
    )

    # Unfold…
    scheduler = SequentialScheduler(setup)
    results   = scheduler.unfold()

    # Collect…
    evaluator = Evaluator(setup.goal, setup.solution, results)
    scores    = evaluator.collect()

    return scores

