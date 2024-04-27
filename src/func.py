import backoff
import inspect
import json
import os
import numpy as np

from tqdm import tqdm
from time import sleep, time
from openai import RateLimitError
from typing import List, Callable, Optional

from symai import Expression, GlobalSymbolPrimitive
from symai.functional import EngineRepository
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine
from symai.backend.engines.index.engine_vectordb import VectorDBIndexEngine
from symai.collect.stats import Aggregator

from src.engines.engine_mockup import MockupEngine
from src.engines.engine_llamacpp import LLaMACppClientEngine
from src.engines.engine_google_vertex import GoogleGeminiEngine
from src.utils import measure, embedding_mean, cross_validation_score
from src.evals import eval_in_context_associations
from src.evals import eval_multimodal_bindings
from src.evals import eval_program_synthesis
from src.evals import eval_logic_components
from src.evals import eval_computation_graphs


BENCHMARK_NAME_MAPPING = {
    'eval_in_context_associations': 'Associations',
    'eval_multimodal_bindings': 'Modality',
    'eval_program_synthesis': 'Code',
    'eval_logic_components': 'Logic',
    'eval_computation_graphs': 'Graphs'
}


MODEL_NAME_MAPPING = {
    'gpt4': 'GPT-4 Turbo',
    'gpt3.5': 'GPT-3.5 Turbo',
    'gemini': 'Gemini 1.0 Pro',
    'gemini1.5': 'Gemini 1.5 Pro',
    'llama': 'LlaMA 2 13B',
    'mistral': 'Mistral 7B',
    'zephyr': 'Zephyr 7B',
    'llama3_8B': 'LlaMA 3 8B',
    'llama3_70B': 'LlaMA 3 70B',
    'random': 'Random'
}


DUMMY_DATA = {
    f"{MODEL_NAME_MAPPING['gpt3.5']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.8},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.6},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.7},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.67},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.7}
    },
    f"{MODEL_NAME_MAPPING['gpt4']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.95},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.85},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.8},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.85},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.9}
    },
    f"{MODEL_NAME_MAPPING['gemini']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.9},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.89},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.78},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.75},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.7}
    },
    f"{MODEL_NAME_MAPPING['llama']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.6},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.45},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.4},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.3},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.45}
    },
    f"{MODEL_NAME_MAPPING['mistral']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.67},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.5},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.45},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.4},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.3}
    },
    f"{MODEL_NAME_MAPPING['zephyr']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.7},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.6},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.5},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.43},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.36}
    },
    f"{MODEL_NAME_MAPPING['random']}": {
        f"{BENCHMARK_NAME_MAPPING['eval_in_context_associations']}": {"performance": 0.11},
        f"{BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']}": {"performance": 0.05},
        f"{BENCHMARK_NAME_MAPPING['eval_program_synthesis']}": {"performance": 0.1},
        f"{BENCHMARK_NAME_MAPPING['eval_logic_components']}": {"performance": 0.09},
        f"{BENCHMARK_NAME_MAPPING['eval_computation_graphs']}": {"performance": 0.1}
    }
}


def load_test_functions(module, prefix='test_'):
    """
    Load all test functions from a given module that start with a specific prefix.

    Args:
        module: The module from which to load test functions.
        prefix: The prefix that test function names should start with (default is 'test_').

    Returns:
        A list of tuples, where each tuple contains the function name and the function object.
    """
    test_functions = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith(prefix):
            test_functions.append((name, func))

    return test_functions


# Define a function that raises an exception without any remedy
def except_remedy(error, *args, **kwargs):
    raise Exception(error) # raise the error without any remedy


class EvaluateBenchmark(Expression):
    def __init__(self, eval_in_context_associations: Optional[List[Callable]] = None,
                       eval_multimodal_bindings: Optional[List[Callable]] = None,
                       eval_program_synthesis: Optional[List[Callable]] = None,
                       eval_logic_components: Optional[List[Callable]] = None,
                       eval_computation_graphs: Optional[List[Callable]] = None,
                       verbose: bool = False,
                       **kwargs):
        super().__init__(**kwargs)
        self.eval_in_context_associations = eval_in_context_associations
        self.eval_multimodal_bindings     = eval_multimodal_bindings
        self.eval_program_synthesis       = eval_program_synthesis
        self.eval_logic_components        = eval_logic_components
        self.eval_computation_graphs      = eval_computation_graphs
        self.verbose                      = verbose
        self.aggregator                   = Aggregator()
        # set global primitives for symbolic AI
        GlobalSymbolPrimitive('measure', measure)
        GlobalSymbolPrimitive('mean', embedding_mean)
        GlobalSymbolPrimitive('cvs', cross_validation_score)
        # Register index engine globally for all Symbols
        EngineRepository.register('index', VectorDBIndexEngine(index_name='dataindex', index_dims=768, index_top_k=5))
        # Register embeddings engine globally for all Symbols from plugin
        EngineRepository.register_from_plugin('embedding', plugin='ExtensityAI/embeddings', kwargs={'model': 'all-mpnet-base-v2'}, allow_engine_override=True)

    def prepare(self, experiment, seed, config, results, type):
        # Set the engine error rate exception if necessary
        rate_exception = None
        engine         = None

        # Set the engine configuration
        if 'gpt' in experiment:
            rate_exception = RateLimitError
            # initialize the engine
            engine = GPTXChatEngine(api_key=config[experiment]['api_key'],
                                    model=config[experiment]['model'])
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
        elif 'gemini' in experiment:
            # initialize the engine
            engine = GoogleGeminiEngine(api_key=config[experiment]['api_key'],
                                        model=config[experiment]['model'])
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
        elif 'llama' in experiment:
            # initialize the engine
            engine = LLaMACppClientEngine(host=config[experiment]['host'],
                                          port=config[experiment]['port'])
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
        elif experiment == 'zephyr':
            # initialize the engine
            engine = LLaMACppClientEngine(host=config[experiment]['host'],
                                          port=config[experiment]['port'])
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
        elif experiment == 'mistral':
            # initialize the engine
            engine = LLaMACppClientEngine(host=config[experiment]['host'],
                                          port=config[experiment]['port'])
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
        elif experiment == 'random':
            # initialize the engine
            engine = MockupEngine(verbose=self.verbose)
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)

        assert engine is not None, f'Engine {experiment} not found!'
        # Set the engine configuration
        experiment = MODEL_NAME_MAPPING[experiment]
        if not results[experiment][type]['engine']:
            results[experiment][type]['engine'] = str(engine.__class__)

        # Check if engine is available and send request to avoid cold start of engine
        # Set the engine configuration
        EngineRepository.command('neurosymbolic', seed=seed)
        EngineRepository.command('neurosymbolic', except_remedy=except_remedy)

        return engine, rate_exception

    def evaluate_experiment(self, experiments, evals, n_runs, seeds, config, results, type='eval_in_context_associations', tests=None):
        type = BENCHMARK_NAME_MAPPING[type]

        experiment_cnt = 0
        for experiment in experiments:
            experiment = MODEL_NAME_MAPPING[experiment]
            results[experiment][type]  = {}
            results[experiment][type]['scores']     = []
            results[experiment][type]['success']    = []
            results[experiment][type]['total_runs'] = n_runs * len(seeds)
            results[experiment][type]['total_time'] = []
            results[experiment][type]['run_list']   = []
            results[experiment][type]['engine']     = None

        assert len(evals) > 0, 'No test functions found!'
        assert len(seeds) > 0, 'No seeds found!'
        assert len(experiments) > 0, 'No experiments found!'
        assert n_runs > 0, 'Number of runs must be greater than 0!'

        print(f'Running {len(evals)} test(s) for {n_runs} run(s), each with {len(seeds)} seed(s) per experiment.')
        # We alter between the test functions and the seeds per experiment since this creates a natural API cooldown between runs
        total_experiments = len(evals) * n_runs * len(seeds) * len(experiments)
        # set tqdm progress bar
        progress = tqdm(total=total_experiments, desc=f'Running {type} benchmark')
        for fun_name, test_func in evals:
            # Skip tests that are not in the list of tests to run
            if tests is not None and fun_name not in tests:
                continue
            # Run the test function for each seed
            for seed in seeds:
                # Run the test function
                for r in range(n_runs):
                    # Evaluate for each engine
                    for experiment in experiments:
                        # Prepare the engine
                        engine, rate_exception = self.prepare(experiment, seed, config, results, type)
                        # set experiment name
                        experiment = MODEL_NAME_MAPPING[experiment]
                        # is mock test
                        is_mock = False
                        # Run the test function
                        # Use exponential backoff to handle API rate limit exceptions
                        @backoff.on_exception(backoff.expo, rate_exception, max_time=60)
                        def run_with_backoff(*args, **kwargs):
                            nonlocal is_mock
                            start_time      = time()  # Start timing
                            aggregator = self.aggregator[experiment][type][fun_name]
                            try:
                                res, info = test_func(aggregator, *args, **kwargs)
                            except Exception as e:
                                print('EVAL FAILURE:', fun_name, e) # Ignore exceptions and count as a failure
                                return False, 0.0, [0.0]
                            finally:
                                sleep(0.05) # Sleep for 50ms for min. API cooldown
                            end_time = time()  # End timing
                            elapsed_time = end_time - start_time
                            entry = {
                                'run': r,
                                'test': fun_name,
                                'seed': seed,
                                'time': elapsed_time,
                                'success': res,
                                'scores': info['scores']
                            }
                            # Check if the test function is a mock test
                            if 'mock' in info and info['mock']:
                                is_mock = True
                            else:
                                results[experiment][type]['run_list'].append(entry)
                            return res, elapsed_time, info['scores']
                        # Run the test function with backoff
                        result, elapsed_time, scores = run_with_backoff()
                        # Check if the test function is a mock test
                        if not is_mock:
                            experiment_cnt += 1
                            results[experiment][type]['total_time'].append(elapsed_time)     # Accumulate time
                            # Check if the test function passed
                            results[experiment][type]['scores'].append(np.mean(scores))      # Count scoring
                            if result:
                                results[experiment][type]['success'].append(1.0)    # Count success
                        # Update progress bar
                        progress.update(1)
                        # print progress
                        mean_score = np.mean(results[experiment][type]['scores']) if experiment_cnt > 0 else 0.0
                        progress.set_postfix({f'{experiment}: mean score': mean_score, 'time': elapsed_time})

        # Calculate the average scoring for associations
        for experiment in experiments:
            experiment = MODEL_NAME_MAPPING[experiment]
            results[experiment][type] = {
                'performance':  np.mean(results[experiment][type]['scores'])     if experiment_cnt > 0 else 0.0,
                'pass_rate':    np.mean(results[experiment][type]['success'])    if experiment_cnt > 0 else 0.0,
                'average_time': np.mean(results[experiment][type]['total_time']) if experiment_cnt > 0 else 0.0,
                'unique_tests': experiment_cnt,
                'seeds':        seeds,
                'runs':         results[experiment][type]['run_list']
            }

        # partial save of current state of results to disk
        os.makedirs('results', exist_ok=True)
        # get only the current type results
        type_results = {experiment: results[experiment][type] for experiment in results}
        with open(f'results/{type}_results.json', 'w') as f:
            json.dump(type_results, f, indent=2)

    def forward(self, experiments=["gpt4", "llama", "gpt3.5", "zephyr", "gemini", "mistral", "random"], n_runs=3, seeds=[42, 77, 97], dummy=False, tests=None):
        # This dictionary will now hold the scoring for each test type
        results = {}
        for experiment in experiments:
            experiment = MODEL_NAME_MAPPING[experiment]
            results[experiment] = {}

        # Load json config file
        with open('config.json', 'r') as f:
            config = json.load(f)

        # If dummy is True, return dummy data
        if dummy:
            return DUMMY_DATA

        # Evaluate in-context learning associations
        if self.eval_in_context_associations:
            self.evaluate_experiment(experiments, self.eval_in_context_associations, n_runs, seeds, config, results, type='eval_in_context_associations', tests=tests)

        # Evaluate multimodal bindings
        if self.eval_multimodal_bindings:
            self.evaluate_experiment(experiments, self.eval_multimodal_bindings, n_runs, seeds, config, results, type='eval_multimodal_bindings', tests=tests)

        # Evaluate program synthesis
        if self.eval_program_synthesis:
            self.evaluate_experiment(experiments, self.eval_program_synthesis, n_runs, seeds, config, results, type='eval_program_synthesis', tests=tests)

        # Evaluate components
        if self.eval_logic_components:
            self.evaluate_experiment(experiments, self.eval_logic_components, n_runs, seeds, config, results, type='eval_logic_components', tests=tests)

        # Evaluate computation graphs
        if self.eval_computation_graphs:
            self.evaluate_experiment(experiments, self.eval_computation_graphs, n_runs, seeds, config, results, type='eval_computation_graphs', tests=tests)

        # save the results file to disk
        os.makedirs('results', exist_ok=True)
        with open(f'results/total_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        # save the score and vector aggregations to disk
        self.aggregator.save(f'results/aggregation.json')

        return results


def run(args):
    # Create list of test functions
    in_context_associations_tests = load_test_functions(eval_in_context_associations) if args.context_associations or args.all else None
    multimodal_bindings_tests     = load_test_functions(eval_multimodal_bindings) if args.multimodal_bindings or args.all else None
    program_synthesis_tests       = load_test_functions(eval_program_synthesis) if args.program_synthesis or args.all else None
    logic_components_tests        = load_test_functions(eval_logic_components) if args.logic_components or args.all else None
    computation_graphs_tests      = load_test_functions(eval_computation_graphs) if args.computation_graphs or args.all else None

    # Instantiate benchmarker
    benchmarker = EvaluateBenchmark(
        eval_in_context_associations=in_context_associations_tests,
        eval_multimodal_bindings=multimodal_bindings_tests,
        eval_program_synthesis=program_synthesis_tests,
        eval_logic_components=logic_components_tests,
        eval_computation_graphs=computation_graphs_tests,
        verbose=args.verbose
    )

    # Run benchmark
    seeds  = [42, 18, 97, 3, 200, 32, 815, 6] if not args.seeds or 'all' in args.seeds else args.seeds
    models = ["gpt4", "llama", "gpt3.5", "zephyr", "gemini", "mistral", "llama3_8B", "llama3_70B", "random"] if not args.models or 'all' in args.models else args.models
    tests  = None if not args.tests or 'all' in args.tests else args.tests
    benchmark_results = benchmarker(experiments=models,
                                    n_runs=1,
                                    seeds=seeds,
                                    dummy=args.dummy,
                                    tests=tests)

    # Print benchmark results
    print("Results:", benchmark_results)

    return benchmark_results
