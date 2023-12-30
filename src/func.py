import inspect
import backoff
import json
import numpy as np

from tqdm import tqdm
from time import sleep, time
from openai import RateLimitError

from symai import Symbol, Expression
from symai.components import TokenTracker
from typing import List, Callable, Optional
from symai.functional import EngineRepository
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import GPTXChatEngine

from src.evals import eval_in_context_associations


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
    raise error # raise the error without any remedy


class EvaluateBenchmark(Expression):
    def __init__(self, eval_in_context_associations: Optional[List[Callable]] = None,
                       eval_multimodal_bindings: Optional[List[Callable]] = None,
                       eval_program_synthesis: Optional[List[Callable]] = None,
                       eval_components: Optional[List[Callable]] = None,
                       eval_computation_graphs: Optional[List[Callable]] = None):
        super().__init__()
        self.eval_associations = eval_in_context_associations
        self.eval_multimodal_bindings = eval_multimodal_bindings
        self.eval_program_synthesis = eval_program_synthesis
        self.eval_components = eval_components
        self.eval_computation_graphs = eval_computation_graphs

    def prepare(self, experiment, seed, config, results, type):
        # Set the engine error rate exception if necessary
        rate_exception = None
        engine         = None
        if experiment == 'gpt4' or experiment == 'gpt3.5':
            rate_exception = RateLimitError
            # initialize the engine
            engine = GPTXChatEngine(api_key=config[experiment]['api_key'],
                                    model=config[experiment]['model'])
            EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
            # Set the engine configuration
            if not results[experiment][type]['engine']:
                results[experiment][type]['engine'] = str(engine.__class__)
        elif experiment == 'gemini':
            pass # TODO: Add Gemini engine
        elif experiment == 'lama':
            pass # TODO: Add LAMA engine

        assert engine is not None, f'Engine {experiment} not found!'
        # Check if engine is available and send request to avoid cold start of engine
        # Set the engine configuration
        # TODO: Add more configuration options
        EngineRepository.command('neurosymbolic', seed=seed)
        EngineRepository.command('neurosymbolic', except_remedy=except_remedy)

        return engine, rate_exception

    def evaluate_experiment(self, experiments, evals, n_runs, seeds, config, results, type='eval_associations'):
        for experiment in experiments:
            results[experiment][type] = {}
            results[experiment][type]['scores'] = []
            results[experiment][type]['total_runs'] = n_runs * len(evals) * len(seeds)
            results[experiment][type]['total_time'] = []
            results[experiment][type]['run_list'] = []
            results[experiment][type]['engine'] = None

        print(f'Running {len(evals)} tests for {n_runs} runs, each with {len(seeds)} seeds per experiment.')
        # We alter between the test functions and the seeds per experiment since this creates a natural API cooldown between runs
        for _, test_func in tqdm(evals):
            for seed in seeds:
                # Run the test function
                for r in range(n_runs):
                    # Evaluate for each engine
                    for experiment in experiments:
                        # Prepare the engine
                        engine, rate_exception = self.prepare(experiment, seed, config, results, type)
                        # Run the test function

                        # Use exponential backoff to handle API rate limit exceptions
                        @backoff.on_exception(backoff.expo, rate_exception, max_time=60)
                        def run_with_backoff(*args, **kwargs):
                            start_time = time()  # Start timing
                            try:
                                res, info = test_func(*args, **kwargs)
                            except Exception as e:
                                print('ERROR:', e) # Ignore exceptions and count as a failure
                                return False, 0.0, 0.0
                            finally:
                                sleep(0.05) # Sleep for 50ms for min. API cooldown
                            end_time = time()  # End timing
                            elapsed_time = end_time - start_time
                            results[experiment][type]['run_list'].append(f"RUN#: {r} {test_func.__name__}, Seed: {seed}, Time: {elapsed_time}, Info: {info}")
                            return res, elapsed_time, info['score']
                        # Run the test function with backoff
                        result, elapsed_time, score = run_with_backoff()
                        results[experiment][type]['total_time'].append(elapsed_time)  # Accumulate time
                        # Check if the test function passed
                        if result:
                            results[experiment][type]['scores'].append(score)  # Count scoring

        # Calculate the average scoring for associations
        for experiment in experiments:
            results[experiment][type] = {
                'performance': np.sum(results[experiment][type]['scores']) / results[experiment][type]['total_runs'],
                'average_time': np.mean(results[experiment][type]['total_time']),
                'unique_tests': len(evals),
                'seeds': seeds,
                'runs': results[experiment][type]['run_list']
            }

    def forward(self, experiments=['gpt4', 'gemini', 'lama', 'gpt3.5'], n_runs=3, seeds=[42, 77, 97]):
        # This dictionary will now hold the scoring for each test type
        results = {}
        for experiment in experiments:
            results[experiment] = {}

        # Load json config file
        with open('config.json', 'r') as f:
            config = json.load(f)

        # Evaluate in-context learning associations
        if self.eval_associations:
            self.evaluate_experiment(experiments, self.eval_associations, n_runs, seeds, config, results, type='eval_associations')

        # # Evaluate multimodal bindings
        # if self.eval_multimodal_bindings:
        #     self.evaluate_experiment(experiments, self.eval_multimodal_bindings, n_runs, seeds, config, results, type='eval_multimodal_bindings')

        return results


def run():
    # Create list of test functions
    in_context_associations_tests = load_test_functions(eval_in_context_associations)

    # Instantiate benchmarker
    benchmarker = EvaluateBenchmark(
        eval_in_context_associations=in_context_associations_tests,
    )

    # Run benchmark
    benchmark_results = benchmarker(experiments=['gpt4', 'gpt3.5'],
                                    n_runs=1,
                                    seeds=[42])

    # Print benchmark results
    print("In-context associations results:", benchmark_results)


if __name__ == '__main__':
    run()