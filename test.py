import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.func import run, BENCHMARK_NAME_MAPPING


def parse_args():
    parser = argparse.ArgumentParser(description='Run the benchmark.')
    parser.add_argument('--context_associations', action='store_true', help='Run the in-context associations benchmark.')
    parser.add_argument('--multimodal_bindings', action='store_true', help='Run the multimodal bindings benchmark.')
    parser.add_argument('--program_synthesis', action='store_true', help='Run the program synthesis benchmark.')
    parser.add_argument('--components', action='store_true', help='Run the components benchmark.')
    parser.add_argument('--computation_graphs', action='store_true', help='Run the computation graphs benchmark.')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks.')
    parser.add_argument('--dummy', action='store_true', help='Run the dummy benchmark.')
    return parser.parse_args()


LATEX_TEMPLATE = """
\\begin{{figure*}}[ht]
  \\centering
  \\begin{{minipage}}{{.6\\textwidth}}
    \\centering
    \\begin{{adjustbox}}{{max width=\\linewidth}}
    \\begin{{tabular}}{{lcccccc}}
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


def create_latex_result(data):
    # Define the directory and file name
    directory = 'tmp'
    # make sure the directory exists
    os.makedirs(directory, exist_ok=True)
    filename = 'benchmark_results.tex'
    filepath = os.path.join(directory, filename)

    # Gather the model names
    model_names = " & ".join(key for key in data.keys())

    # Initialize the total scores
    total_scores = {model: 0.0 for model in data.keys()}

    # Prepare table content
    benchmark_rows = {bench_name: "" for bench_name in BENCHMARK_NAME_MAPPING.values()}
    for bench_name in BENCHMARK_NAME_MAPPING.values():
        if bench_name not in str(list(data.values())):
            print(f"Skipping benchmark because not all results are computed. Did not find `{bench_name}` in `{data.keys()}`")
            return
        # Initialize list to keep the scores for this benchmark to find the best model
        scores = [(model, values[bench_name]['performance']) for model, values in data.items()]
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
        benchmark_components_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_components']],
        benchmark_computation_graphs_row=benchmark_rows[BENCHMARK_NAME_MAPPING['eval_computation_graphs']],
        total_row='Total' + total_values
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
    values = [[v['performance'] for v in sublist] for sublist in values]
    values = np.array(values)

    # Create a radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = np.concatenate((values, values[:,[0]]), axis=1)  # Repeat the first value to close the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    sns.set_theme(context='paper', style='whitegrid')

    def add_to_radar(model, color):
        ax.plot(angles, model, color=color, linewidth=2, label=model_name)
        ax.fill(angles, model, color=color, alpha=0.25)

    # Add each model to the radar chart
    for values, model_name in zip(values, models):
        add_to_radar(values, ax._get_lines.get_next_color())

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
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15), fontsize=label_font_size)

    # Set tight layout
    plt.tight_layout()

    # Save as PDF
    plt.savefig("tmp/benchmark_comparison_chart.pdf", format="pdf")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    results = run(args)
    create_latex_result(results)
    create_plot(results)

