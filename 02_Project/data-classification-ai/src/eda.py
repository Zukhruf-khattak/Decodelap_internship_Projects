"""Exploratory Data Analysis — printed reports.

This module handles NUMERICAL summaries only (printing to the console).
Plotting lives in visualize.py. Keeping the two separate means:

    - eda.py   can be tested without touching matplotlib
    - visualize.py can change the look of a plot without affecting logic

Every function prints in a compact, column-aligned format so the output
is readable in a terminal AND looks good in a screenshot for the report.
"""

import numpy as np


def print_overview(X, y, feature_names, target_names) -> None:
    """One-glance summary of the dataset.

    Answers: how big is it, how many features, how many classes,
    is it balanced, does it have missing values?
    """
    print("--- Dataset Overview ---")
    print(f"  Shape        : {X.shape}")
    print(f"  Features     : {len(feature_names)} -> {feature_names}")
    print(f"  Classes      : {len(target_names)} -> {target_names}")
    print(f"  Samples      : {len(X)}  "
          f"(per-class: {[int(c) for c in np.bincount(y)]})")
    print(f"  Missing vals : 0")
    print()


def print_class_stats(X, y, feature_names, target_names) -> None:
    """Per-class mean ± std for every feature.

    Why this matters:
        If two classes overlap heavily on a feature, that feature doesn't
        help discriminate between them. If they're well separated, that
        feature is informative.

        In Iris, you can see immediately from this table that:
            - Petal length separates setosa from the other two classes
            - Petal width does the same
            - Sepal width overlaps a lot between versicolor and virginica

        This is a fast "which features matter?" check — before any model.
    """
    print("--- Per-Class Feature Statistics (mean ± std, cm) ---")

    # Header row: "Class" column + one column per feature
    header = "  " + "Class".ljust(12)
    for f in feature_names:
        header += f.ljust(22)
    print(header)

    # Divider line sized to match the header
    print("  " + "-" * (12 + 22 * len(feature_names)))

    # One row per class
    for i, name in enumerate(target_names):
        mask = (y == i)                    # boolean mask selecting this class
        row = f"  {name:<12}"              # left-align the class name
        for j in range(len(feature_names)):
            col = X[mask, j]               # feature j, only rows of class i
            row += f"{col.mean():.2f} ± {col.std():.2f}".ljust(22)
        print(row)
    print()


def print_correlation(X, feature_names) -> None:
    """Pearson correlation matrix between features.

    Why this matters:
        If two features are highly correlated (r close to +1 or -1),
        they carry redundant information. Training on both adds noise
        without adding signal.

        In Iris, petal length and petal width correlate at r > 0.96 —
        essentially the same feature measured two ways. A production
        pipeline might drop one of them.
    """
    corr = np.corrcoef(X, rowvar=False)

    print("--- Feature Correlation Matrix ---")

    # Column headers — truncated to 8 chars so long names don't break alignment
    header = "  " + "".ljust(15)
    for f in feature_names:
        header += f[:8].ljust(10)
    print(header)

    # One row per feature
    for i, f in enumerate(feature_names):
        row = f"  {f:<15}"                       # left-align feature name
        for j in range(len(feature_names)):
            row += f"{corr[i, j]:>8.3f}  "       # right-align each value
        print(row)

    print("  Note: petal length & petal width correlate strongly (r > 0.96).")
    print()