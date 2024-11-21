import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def plot_entropy_vs_accuracy_bar(entropy_dict, accuracy_dict, bins=3, bin_labels=None):
    """
    Plots the distribution of correct and incorrect answers across entropy bins side by side using Seaborn.
    
    Parameters:
        entropy_no_augment_dict (dict): Dictionary where keys are question indices and values are entropy values.
        accuracy_no_augment_dict (dict): Dictionary where keys are question indices and values are booleans (True/False) indicating accuracy.
        bins (int or list): Number of bins or list of bin edges for binning entropy values.
        bin_labels (list, optional): Labels for the bins. If not provided, labels are auto-generated.
    """


    # Create a DataFrame from the dictionaries
    df = pd.DataFrame({
        'entropy': pd.Series(entropy_dict),
        'accuracy': pd.Series(accuracy_dict)
    })

    # Remove any entries with missing data
    df = df.dropna()

    if isinstance(bins, int):
        max_entropy = df['entropy'].max()
        bins = [0] + list(pd.interval_range(start=0, end=max_entropy, freq=max_entropy / bins).right)
    bins = [b - 1e-10 if i == 0 else b for i, b in enumerate(bins)]

    # Bin the entropy values
    df['entropy_bin'] = pd.cut(df['entropy'], bins=bins, labels=bin_labels)

    # Map accuracy boolean to 'Correct' and 'Incorrect'
    df['accuracy_label'] = df['accuracy'].map({True: 'Correct', False: 'Incorrect','uncertain': 'Uncertain'})

    # Prepare the data for plotting
    counts = df.groupby(['entropy_bin', 'accuracy_label']).size().reset_index(name='counts')

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=counts,
        x='entropy_bin',
        y='counts',
        hue='accuracy_label',
        palette='pastel'
    )

    plt.xlabel('Entropy Bins')
    plt.ylabel('Number of Questions')
    plt.title('Distribution of Correct and Incorrect Answers across Entropy Bins')
    plt.legend(title='Accuracy', loc='upper right')

    plt.tight_layout()
    plt.show()