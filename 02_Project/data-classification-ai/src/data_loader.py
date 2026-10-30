"""Phase 1 of IPO — load the Iris dataset.

Priority:
    1. Read from data/iris.csv (committed to the repo).
    2. Fall back to sklearn's bundled copy if the CSV doesn't exist yet.

Both paths return the exact same tuple shape so nothing downstream
(data_loader, preprocessor, model, visualize) cares which source was used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load Iris and return (X, y, feature_names, target_names).

    Returns
    -------
    X : ndarray of shape (150, 4)
        Feature matrix — sepal/petal length and width in cm.
    y : ndarray of shape (150,)
        Integer class labels: 0=setosa, 1=versicolor, 2=virginica.
    feature_names : list[str]
        4 feature column names with " (cm)" stripped for clean printing.
    target_names : list[str]
        ["setosa", "versicolor", "virginica"] in sorted (stable) order.
    """
    csv_path = Path("data/iris.csv")

    if csv_path.exists():
        return _load_from_csv(csv_path)
    return _load_from_sklearn()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _load_from_csv(csv_path: Path):
    """Read the committed CSV into (X, y, feature_names, target_names)."""
    df = pd.read_csv(csv_path)

    # Every column except "species" is a numeric feature
    feature_cols = [c for c in df.columns if c != "species"]

    # Sorted class names -> stable integer labels
    # (setosa=0, versicolor=1, virginica=2, regardless of row order in CSV)
    target_names = sorted(df["species"].unique().tolist())
    label_map = {name: i for i, name in enumerate(target_names)}

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["species"].map(label_map).to_numpy(dtype=int)

    # Strip the " (cm)" suffix so plots and prints stay clean
    feature_names = [c.replace(" (cm)", "") for c in feature_cols]

    return X, y, feature_names, target_names


def _load_from_sklearn():
    """Fallback — use sklearn's bundled copy if the CSV is missing."""
    iris = load_iris()
    return (
        iris.data,
        iris.target,
        [f.replace(" (cm)", "") for f in iris.feature_names],
        list(iris.target_names),
    )