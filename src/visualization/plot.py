"""Plotting functions for analysis and results."""

import matplotlib.pyplot as plt
import seaborn as sns
import os


def setup_style():
    """Set a consistent plot style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})


def save_figure(fig, filename: str, output_dir: str = "outputs/figures"):
    """Save a matplotlib figure to the outputs directory."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Figure saved to {filepath}")


def plot_distribution(df, column: str, title: str = None):
    """Plot the distribution of a single column."""
    setup_style()
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(title or f"Distribution of {column}")
    return fig


def plot_correlation_matrix(df, title: str = "Correlation Matrix"):
    """Plot a heatmap of the correlation matrix."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    return fig


def plot_training_history(history: dict, output_dir: str = None):
    """Plot training loss / metric curves from a history dict.

    history should look like: {"train_loss": [...], "val_loss": [...], "val_f1": [...]}
    """
    setup_style()
    fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 5))
    if len(history) == 1:
        axes = [axes]
    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
    fig.tight_layout()
    if output_dir:
        save_figure(fig, "training_history.png", output_dir)
    return fig


if __name__ == "__main__":
    print("plot module ready")
