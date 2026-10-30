"""All plotting lives here. One function per PNG.

Every function:
    1. Builds a matplotlib figure.
    2. Saves it to the path you passed in.
    3. Closes the figure (prevents memory leaks when generating many plots).

The "Agg" backend means no GUI window pops up — safe for headless
environments (CI, servers, or running scripts in VS Code terminal).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Color palette (consistent across all plots)
PALETTE = {
    "setosa":     "#1f3a5f",
    "versicolor": "#3b6ea5",
    "virginica":  "#e07020",
}
BAR_COLORS = ["#1f3a5f", "#3b6ea5", "#e07020"]


# ======================================================================
# EDA plots
# ======================================================================

def plot_class_balance(y, target_names, output_path):
    """Bar chart of samples per class. Confirms balance at a glance."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = np.bincount(y)
    bars = ax.bar(target_names, counts, color=BAR_COLORS,
                  edgecolor="black", linewidth=0.8)

    # Annotate each bar with its count
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 1, str(int(c)),
                ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Sample count")
    ax.set_title("Class Balance — Iris is 50/50/50", fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pairplot(X, y, feature_names, target_names, output_path):
    """Pairplot: all 6 feature pairs, colored by species."""
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]

    g = sns.pairplot(
        df, hue="species", diag_kind="hist", palette=PALETTE,
        plot_kws={"alpha": 0.7, "s": 30}, height=2.0,
    )
    g.fig.suptitle("Feature Pairplot by Species", fontweight="bold", y=1.02)
    g.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_correlation(X, feature_names, output_path):
    """Heatmap of Pearson correlation between features."""
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdBu_r",
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_features_by_class(X, y, feature_names, target_names, output_path):
    """2x2 grid of boxplots — one per feature, split by species."""
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[i] for i in y]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feat in zip(axes.flat, feature_names):
        sns.boxplot(data=df, x="species", y=feat, ax=ax, palette=BAR_COLORS)
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Feature Distributions by Species",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()


# ======================================================================
# Model plots
# ======================================================================

def plot_elbow(k_range, errors, best_k, output_path):
    """Error-rate vs K, with the optimal K circled — mirrors PDF page 12."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # The error curve
    ax.plot(k_range, errors, "-o", color="#1f3a5f",
            markersize=5, linewidth=1.6, zorder=2)

    # Highlight the elbow in red
    best_error = errors[best_k - k_range[0]]  # index into the errors list
    ax.scatter([best_k], [best_error],
               s=700, facecolors="none", edgecolors="red",
               linewidth=3, zorder=4)
    ax.annotate(
        "THE ELBOW\n(Optimal K)",
        xy=(best_k, best_error),
        xytext=(best_k + 4, best_error + max(errors) * 0.25),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=11, fontweight="bold", ha="left",
    )

    # Overfitting / underfitting labels
    ax.text(2, max(errors) * 0.85,
            "K=1\n(Noise / Overfitting)",
            fontsize=9, color="gray", style="italic")
    ax.text(max(k_range) - 1, max(errors) * 0.85,
            f"K={max(k_range)}\n(Generic / Underfitting)",
            fontsize=9, color="gray", style="italic", ha="right")

    ax.set_xlabel("K Value", fontsize=11)
    ax.set_ylabel("Error Rate (1 - CV accuracy)", fontsize=11)
    ax.set_title("Tuning the Engine: Choosing K via 5-Fold CV",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(min(k_range), max(k_range) + 1, 2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_algorithm_comparison(results, output_path):
    """Horizontal grouped bars: CV accuracy and macro-F1 per algorithm."""
    names = [r["name"] for r in results]
    acc_means = [r["acc_mean"] for r in results]
    f1_means = [r["f1_mean"] for r in results]

    y_pos = np.arange(len(names))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.barh(y_pos - height / 2, acc_means, height,
                 label="CV accuracy", color="#1f3a5f")
    b2 = ax.barh(y_pos + height / 2, f1_means, height,
                 label="CV macro-F1", color="#e07020")

    # Value labels at the end of each bar
    for bars in (b1, b2):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0.85, 1.02)
    ax.set_xlabel("Score (5-fold CV)")
    ax.set_title("Algorithm Comparison — Same Pipeline, Same Folds",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()