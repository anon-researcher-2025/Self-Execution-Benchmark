import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_f1_histogram(series, title):
    """Plots a histogram of the F1 scores."""
    plt.figure(figsize=(8, 6))
    plt.hist(series.dropna(), bins=20)
    plt.title(title)
    plt.xlabel('F1 Score')
    plt.ylabel('Number of Cases')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_f1_kde(series, title):
    """Plots a KDE distribution of F1 scores with mean and rug plot."""
    mean_f1 = series.dropna().mean()
    plt.figure(figsize=(8, 6))
    sns.kdeplot(series.dropna(), color="green", fill=True, alpha=0.5, label='F1 Score Density', clip=(0, 1))
    sns.rugplot(series.dropna(), color="black", height=0.05)
    plt.axvline(mean_f1, color='red', linestyle='dashed', linewidth=2, label=f'Mean F1 Score: {mean_f1:.2f}')
    plt.xlim(left=0, right=1) # F1 score is between 0 and 1
    plt.title(title)
    plt.xlabel('F1 Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def plot_overlaid_kde(df, title):
    """Plots overlaid KDE distributions for precision, recall, and F1 score."""
    mean_precision = df['precision'].dropna().mean()
    mean_recall = df['recall'].dropna().mean()
    mean_f1 = df['f1_score'].dropna().mean()

    plt.figure(figsize=(10, 7))

    sns.kdeplot(df['precision'].dropna(), color="skyblue", fill=True, alpha=0.5,
                label=f'Precision (Mean: {mean_precision:.2f})', clip=(0, 1))
    sns.kdeplot(df['recall'].dropna(), color="lightcoral", fill=True, alpha=0.5,
                label=f'Recall (Mean: {mean_recall:.2f})', clip=(0, 1))
    sns.kdeplot(df['f1_score'].dropna(), color="lightgreen", fill=True, alpha=0.5,
                label=f'F1 Score (Mean: {mean_f1:.2f})', clip=(0, 1))

    plt.axvline(mean_precision, color='blue', linestyle='dashed', linewidth=1.5, label=f'_nolegend_') # Hide from legend
    plt.axvline(mean_recall, color='red', linestyle='dashed', linewidth=1.5, label=f'_nolegend_')
    plt.axvline(mean_f1, color='darkgreen', linestyle='dashed', linewidth=1.5, label=f'_nolegend_')

    plt.xlim(left=0, right=1)
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
