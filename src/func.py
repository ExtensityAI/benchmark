import inspect
import backoff

from time import sleep, time
from openai import RateLimitError

from symai import Symbol, Expression
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

    def forward(self, experiments=['gpt4', 'gemini', 'lama'], n_runs=3, seeds=[42, 77, 97]):
        # This dictionary will now hold the success rate for each test type
        success_rates = {}

        # Define a function that raises an exception without any remedy
        def except_remedy(error, *args, **kwargs):
            raise error # raise the error without any remedy

        # Evaluate for each engine
        for experiment in experiments:
            # Set the engine error rate exception if necessary
            rate_exception = None
            engine         = None
            if experiment == 'gpt4':
                rate_exception = RateLimitError
                # initialize the engine
                engine = GPTXChatEngine()
                EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
            elif experiment == 'gemini':
                pass # TODO: Add Gemini engine
            elif experiment == 'lama':
                pass # TODO: Add LAMA engine

            assert engine is not None, f'Engine {experiment} not found!'
            # Check if engine is available and send request to avoid cold start of engine
            Symbol(1) == 1 # ping the engine

            # Evaluate in-context learning associations
            successes  = 0
            total_runs = n_runs * len(self.eval_associations) * len(seeds)
            total_time = 0.0
            if self.eval_associations:
                for _, test_func in self.eval_associations:
                    for seed in seeds:
                        # Set the engine configuration
                        # TODO: Add more configuration options
                        EngineRepository.command('neurosymbolic', seed=seed)
                        EngineRepository.command('neurosymbolic', except_remedy=except_remedy)

                        # Run the test function
                        for _ in range(n_runs):
                            try:
                                # Use exponential backoff to handle API rate limit exceptions
                                @backoff.on_exception(backoff.expo, rate_exception, max_time=60)
                                def run_with_backoff(*args, **kwargs):
                                    start_time = time()  # Start timing
                                    res = test_func(*args, **kwargs)
                                    end_time = time()  # End timing
                                    return res, end_time - start_time
                                # Run the test function with backoff
                                result, elapsed_time = run_with_backoff()
                                total_time += elapsed_time  # Accumulate time
                                # Check if the test function passed
                                if result:
                                    successes += 1  # Count successful outcomes
                            except Exception as e:
                                print('ERROR:', e) # Ignore exceptions and count as a failure
                            finally:
                                sleep(0.1) # Sleep for 100ms for API cooldown
            # Calculate the average success rate for associations
            success_rates['eval_associations'] = {
                'success_rate': successes / total_runs,
                'average_time': total_time / total_runs,
                'total_runs': total_runs,
                'unique_tests': len(self.eval_associations),
                'seeds': seeds,
                'experiment': experiment,
                'engine': str(engine.__class__)
            }

        return success_rates


def run():
    # Create list of test functions
    in_context_associations_tests = load_test_functions(eval_in_context_associations)

    # Instantiate benchmarker
    benchmarker = EvaluateBenchmark(
        eval_in_context_associations=in_context_associations_tests,
    )

    # Run benchmark
    benchmark_results = benchmarker(experiments=['gpt4'],
                                    n_runs=1,
                                    seeds=[42])

    # Print benchmark results
    print("In-context associations average success rate:", benchmark_results['eval_associations'])


if __name__ == '__main__':
    run()