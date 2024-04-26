import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.func import run, BENCHMARK_NAME_MAPPING, MODEL_NAME_MAPPING


def parse_args():
    parser = argparse.ArgumentParser(description='Run the benchmark.')
    parser.add_argument('--context_associations', help='Run the in-context associations benchmark.', action='store_true')
    parser.add_argument('--multimodal_bindings',  help='Run the multimodal bindings benchmark.',     action='store_true')
    parser.add_argument('--program_synthesis',    help='Run the program synthesis benchmark.',       action='store_true')
    parser.add_argument('--logic_components',     help='Run the logic components benchmark.',        action='store_true')
    parser.add_argument('--computation_graphs',   help='Run the computation graphs benchmark.',      action='store_true')
    parser.add_argument('--all',                  help='Run all benchmarks.',                        action='store_true')
    parser.add_argument('--dummy',                help='Run the dummy benchmark.',                   action='store_true')
    parser.add_argument('--verbose',              help='Print additional information.',              action='store_true')
    parser.add_argument('--models',               help='Run the specified models.',                  default=['all'],       type=str, nargs='+')
    parser.add_argument('--seeds',                help='Run the specified seeds.',                   default=['all'],       type=int, nargs='+')
    parser.add_argument('--tests',                help='Run only specific tests.',                   default=['all'],       type=str, nargs='+')
    parser.add_argument('--plot',                 help='Plot the results.',                                                 type=str)
    return parser.parse_args()


LATEX_TEMPLATE = """
\\begin{{figure*}}[ht]
  \\centering
  \\begin{{minipage}}{{.6\\textwidth}}
    \\centering
    \\begin{{adjustbox}}{{max width=\\linewidth}}
    \\begin{{tabular}}{{lccccccc}}
      \\toprule
      \\textbf{{Benchmarks}} &  {model_names} \\\\
      \\midrule
      {benchmark_in_context_association_row} \\\\
      {benchmark_multimodality_row} \\\\
      {benchmark_program_synthesis_row} \\\\
      {benchmark_components_row} \\\\
      {benchmark_computation_graphs_row} \\\\
      \\midrule
      \\textbf{{Total}} & {total_row} \\\\
      \\bottomrule
    \\end{{tabular}}
    \\end{{adjustbox}}
    \\label{{tab:benchmark_results}}
  \\end{{minipage}}%
  ~
  \\begin{{minipage}}{{.4\\textwidth}}
    \\centering
    \\begin{{adjustbox}}{{max width=\\linewidth}}
    \\includegraphics[width=\\linewidth]{{images/benchmark_comparison_chart.pdf}}
    \\end{{adjustbox}}
    \\label{{fig:spider_plot}}
  \\end{{minipage}}
  \\caption{{Placeholder for performance benchmarks and comparison chart for various models.}}
  \\label{{fig:my_label}}
\\end{{figure*}}
"""


def sort_by_name(model):
    if 'GPT-4' in model:
        return 0
    elif 'GPT-3.5' in model:
        return 1
    elif 'Gemini' in model:
        return 2
    elif 'LlaMA' in model:
        return 3
    elif 'Mistral' in model:
        return 4
    elif 'Zephyr' in model:
        return 5
    elif 'Random' in model:
        return 6
    else:
        return 7


def sort_items_by_name(model):
    return sort_by_name(model[0])


remap_name = {
    'GPT-4 Turbo': 'GPT-4',
    'GPT-3.5 Turbo': 'GPT-3.5',
    'Gemini 1.0 Pro': 'Gemini 1.0',
    'Gemini 1.5 Pro': 'Gemini 1.5',
    'LlaMA 2 13B': 'LlaMA 2',
    'LlaMA 3 8B': 'LlaMA 3 8B',
    'LlaMA 3 70B': 'LlaMA 3 70B',
    'Zephyr 7B': 'Zephyr',
    'Mistral 7B': 'Mistral',
    'Random': 'Random'
}


def create_latex_result(data):
    # Define the directory and file name
    directory = 'tmp'
    # make sure the directory exists
    os.makedirs(directory, exist_ok=True)
    filename = 'benchmark_results.tex'
    filepath = os.path.join(directory, filename)

    # Gather the model names
    data_model_names = list(data.keys())
    # Sort the models by name
    data_model_names.sort(key=sort_by_name)
    model_names = " & ".join(remap_name[key] for key in data_model_names)

    # Initialize the total scores
    total_scores = {model: 0.0 for model in data_model_names}

    # Prepare table content
    benchmark_rows = {bench_name: "" for bench_name in BENCHMARK_NAME_MAPPING.values()}
    for bench_name in BENCHMARK_NAME_MAPPING.values():
        if bench_name not in str(list(data.values())):
            print(f"Skipping benchmark because not all results are computed. Did not find `{bench_name}` in `{data.keys()}`")
            return
        # Initialize list to keep the scores for this benchmark to find the best model
        scores = [(model, np.mean([np.mean(run['scores']) for run in values[bench_name]['runs']])) for model, values in data.items()]
        # sort the scores by name following this order: GPT-4, GPT-3.5, Gemini-Pro, LlaMA 2, Mistral, Zephyr, Random
        # write custom sorting function to sort by name
        scores.sort(key=sort_items_by_name)

        best_score = max(scores, key=lambda x: x[1])[1]

        # Create row for the latex table and update the total scores
        row = f"{bench_name}"
        for model, score in scores:
            # Add to the total score
            total_scores[model] += score
            # Format row with best model in bold
            if score == best_score:
                row += f" & \\textbf{{{score:.2f}}}"
            else:
                row += f" & {score:.2f}"
        benchmark_rows[bench_name] = row

    # Compute the average of total scores
    for model in total_scores.keys():
        total_scores[model] /= len(BENCHMARK_NAME_MAPPING)

    # Best total performance in bold
    best_total = max(total_scores.values())
    total_values = " & ".join(f"\\textbf{{{v:.2f}}}" if v == best_total else f"{v:.2f}" for v in total_scores.values())

    # Use the LATEX_TEMPLATE and inject the benchmark rows
    latex_table = LATEX_TEMPLATE.format(
        model_names=model_names,
        benchmark_in_context_association_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_in_context_associations']],
        benchmark_multimodality_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_multimodal_bindings']],
        benchmark_program_synthesis_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_program_synthesis']],
        benchmark_components_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_logic_components']],
        benchmark_computation_graphs_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_computation_graphs']],
        total_row=total_values
    )

    # Print the latex table to the console
    print(latex_table)

    # Save the latex table to a file
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filepath, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {filepath}")


def create_plot(data):
    # Define the categories and models
    categories = list(next(iter(data.values())).keys())  # Assuming all models have the same structure
    models = list(data.keys())
    N = len(categories)

    # Prepare data for plotting
    values = [list(d.values()) for d in data.values()]
    values = [[np.mean([np.mean(run['scores']) for run in v['runs']]) for v in sublist] for sublist in values]
    values = np.array(values)

    # Create a radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = np.concatenate((values, values[:,[0]]), axis=1)  # Repeat the first value to close the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    sns.set_theme(context='paper', style='whitegrid')

    def add_to_radar(values, model_name, color):
        if model_name == MODEL_NAME_MAPPING['random']:
            val = np.max(values) # Use the maximum value to draw a circle for the random model
            angles_circle = np.linspace(0, 2 * np.pi, 100)  # Use 100 points to make a smooth circle
            ax.plot(angles_circle, np.full_like(angles_circle, val), '--', linewidth=2, color=color, label=model_name)
            ax.fill(angles_circle, np.full_like(angles_circle, val), color=color, alpha=0.25)
        else:
            ax.plot(angles, values, color=color, linewidth=2, label=model_name)
            ax.fill(angles, values, color=color, alpha=0.25)

    colors = [ax._get_lines.get_next_color() for _ in range(len(models))]
    zippped = zip(values, models, colors)
    # sort based on name
    zippped = sorted(zippped, key=lambda x: sort_by_name(x[1]))

    # Add each model to the radar chart
    for values, model_name, color in zippped:
        model_name = remap_name[model_name]
        add_to_radar(values, model_name, color)

    # Add labels to the plot with increased label padding
    label_padding = 1.1  # Adjust label padding as needed
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Define font size
    label_font_size = 18  # Choose desired font size
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=label_font_size)
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_position((label_padding, label.get_position()[1]))

    # Increase the font size for the legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2), fontsize=label_font_size)

    # Set tight layout
    plt.tight_layout()

    # Save as PDF
    plt.savefig("tmp/benchmark_comparison_chart.pdf", format="pdf")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    if args.plot is None:
        results = run(args)
    else:
        with open(args.plot, 'r') as f:
            results = json.load(f)
    create_latex_result(results)
    create_plot(results)
