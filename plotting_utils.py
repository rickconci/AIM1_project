import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_entropy_and_accuracy(data):

    data = entropy_ideal_format
    entropy_sums = [0] * len(data[0])

    # Initialize a list to store the count of values for each position
    entropy_counts = [0] * len(data[0])

    # Accumulate sums and counts for each position
    for sublist in data:
        for index, record in enumerate(sublist):
            entropy_sums[index] += record[0]
            entropy_counts[index] += 1

    # Calculate the mean entropy for each position
    mean_entropies = [entropy_sums[i] / entropy_counts[i] for i in range(len(entropy_sums))]

    # Calculate accuracy for each position
    accuracy_sums = [0] * len(data[0])
    accuracy_counts = [0] * len(data[0])

    for sublist in data:
        for index, record in enumerate(sublist):
            accuracy_sums[index] += record[1]  # record[1] contains accuracy (True/False)
            accuracy_counts[index] += 1

    mean_accuracies = [accuracy_sums[i] / accuracy_counts[i] for i in range(len(accuracy_sums))]
    # Set a professional color palette and style
    plt.style.use('seaborn')
    sns.set_palette("deep")

    # Create the figure with better proportions
    plt.figure(figsize=(10, 6))

    # Plot entropy bars with improved aesthetics
    x_labels = ['Low Noise', 'Medium Noise', 'High Noise']

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Entropy bars with gradient and shadow effect
    bars = ax1.bar(x_labels, mean_entropies, 
                color=sns.color_palette("Blues", n_colors=len(mean_entropies)), 
                alpha=0.7, 
                edgecolor='darkblue', 
                linewidth=1.5)

    # Add value labels to the side of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., 0.95*height,
                f'{height:.3f}',
                ha='center', va='bottom', 
                fontweight='bold', 
                color='darkblue',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax1.set_ylabel('Mean Entropy', fontweight='bold', color='darkblue')
    ax1.set_title('GPT4o: Mean Entropy and Accuracy by Question Stem', 
                fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', colors='darkblue')
    ax1.set_xlabel('Amount of Noise Added', fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Second y-axis for accuracy
    ax2 = ax1.twinx()
    accuracy_line = ax2.plot(x_labels, mean_accuracies, 
            color='darkred', 
            marker='o', 
            linewidth=3, 
            markersize=10, 
            label='Accuracy')

    # Add value labels offset from the accuracy points
    for i, acc in enumerate(mean_accuracies):
        ax2.text(i, acc+0.01, f'{acc:.3f}', 
                ha='center', va='bottom', 
                fontweight='bold', 
                color='darkred',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax2.set_ylabel('Accuracy', fontweight='bold', color='darkred')
    ax2.tick_params(axis='y', colors='darkred')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Add subtle grid
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    return fig

def plot_uncertainty_fraction(results_data):
    """
    Plots the fraction of uncertain responses by position.
    """
    y_counts = [0] * len(results_data[0])
    total_counts = [0] * len(results_data[0])

    # Calculate counts
    for sublist in results_data:
        for index, record in enumerate(sublist):
            total_counts[index] += 1
            if record[1] == 'Y':
                y_counts[index] += 1

    fraction_y = [y_counts[i] / total_counts[i] if total_counts[i] > 0 else 0 
                 for i in range(len(y_counts))]

    # Create plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(len(fraction_y)), fraction_y, 
            color='green', marker='o', linestyle='-', 
            linewidth=2, label='Fraction Uncertain')

    plt.title('Fraction of "Uncertain" by Position')
    plt.xlabel('Position in Order')
    plt.ylabel('Fraction Uncertain')
    xticks_labels = ['1 sentence', '2 sentences', 'Half of sentences', 
                     '3/4 of sentences', 'Full question']
    plt.xticks(ticks=range(len(xticks_labels)), labels=xticks_labels, 
               rotation=45, ha="right")
    plt.ylim(0, 0.10)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    return fig

def plot_response_distribution(results_data):
    """
    Creates a grouped bar plot showing the distribution of correct, incorrect, 
    and uncertain responses.
    """
    plt.style.use('seaborn')
    sns.set_palette("husl")

    # Process data
    results = {i: {'correct': 0, 'incorrect': 0, 'uncertain': 0} for i in range(5)}
    total_groups = len(results_data)

    for group in results_data:
        for i, item in enumerate(group):
            if item[1] == 'Y':
                results[i]['uncertain'] += 1
            elif item[0] == 1:
                results[i]['correct'] += 1
            else:
                results[i]['incorrect'] += 1

    # Calculate fractions
    for i in range(5):
        for key in results[i]:
            results[i][key] /= total_groups

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    positions = np.arange(1, 6)
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    hatches = ['', '///', '...']
    bar_width = 0.25

    # Plot bars
    for i, (category, color, hatch) in enumerate(zip(['correct', 'incorrect', 'uncertain'], 
                                                    colors, hatches)):
        data = [results[j][category] for j in range(5)]
        ax.bar(positions + (i-1)*bar_width, data, bar_width, 
               label=category.capitalize(), 
               color=color, edgecolor='black', linewidth=1.5, 
               alpha=0.8, hatch=hatch)

    # Customize plot
    ax.set_xlabel('Amount of Question Stem Given', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fraction of Responses', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Responses by Amount of Question Stem Given', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(['1 sentence', '2 sentences', 'Half of sentences', 
                        '3/4 of sentences', 'Full question'], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Response Type', title_fontsize=12, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)

    fig.patch.set_facecolor('#f0f0f0')
    plt.tight_layout()
    return fig 


def convert_entropy_format(entropy_list, answer_distribution, correct_answers):

    result = []
    for i, (answers, entropy_row) in enumerate(zip(answer_distribution, entropy_list)):
        question_result = []
        for j, (distribution, entropy) in enumerate(zip(answers, entropy_row)):
            most_common = max(set(distribution), key=distribution.count)
            is_correct = most_common == correct_answers[i]
            question_result.append([entropy, is_correct, distribution])
        result.append(question_result)

    # Output the result as a list
    return result