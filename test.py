import argparse
from src.func import run


def parse_args():
    parser = argparse.ArgumentParser(description='Run the benchmark.')
    parser.add_argument('--context_associations', action='store_true', help='Run the in-context associations benchmark.')
    parser.add_argument('--multimodal_bindings', action='store_true', help='Run the multimodal bindings benchmark.')
    parser.add_argument('--program_synthesis', action='store_true', help='Run the program synthesis benchmark.')
    parser.add_argument('--components', action='store_true', help='Run the components benchmark.')
    parser.add_argument('--computation_graphs', action='store_true', help='Run the computation graphs benchmark.')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
