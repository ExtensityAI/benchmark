import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.func import run


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
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize=label_font_size)

    # Save as PDF
    plt.savefig("benchmark_comparison_chart.pdf", format="pdf")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    results = run(args)
    create_plot(results)

