"""Training, tuning, cross-validation, and evaluation.

The single most important design decision in this file:

    Every model is wrapped in sklearn's Pipeline:

        Pipeline([StandardScaler, Classifier])

Why? During cross-validation, the scaler must be re-fit on each
training fold, not once on the whole dataset. If you scale-then-CV,
the scaler has seen the test fold's statistics — that is data
leakage. It silently inflates scores by 1–3% and makes the model
look better than it is on truly unseen data.

By wrapping in Pipeline, sklearn handles this automatically:
    - Fit the scaler on the training folds of that CV split.
    - Transform the held-out fold using that scaler only.
    - Score the held-out fold.

Result: reported scores match what the model will do in production.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ======================================================================
# Building blocks
# ======================================================================

def make_pipeline(estimator) -> Pipeline:
    """Wrap any classifier with a StandardScaler in a single unit.

    Used everywhere in this file — CV, tuning, final training — so
    behavior is identical regardless of which algorithm is plugged in.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


def _make_cv(cv: int = 5) -> StratifiedKFold:
    """Stratified K-fold splitter with a fixed seed.

    Stratified = every fold preserves the 50/50/50 class ratio.
    Fixed seed = results are identical across runs (reproducible).
    """
    return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)


# ======================================================================
# Cross-validation
# ======================================================================

def cv_scores(estimator, X, y, cv: int = 5):
    """Return (accuracy_per_fold, f1_per_fold) for one estimator.

    Each score is a NumPy array of length `cv` — one entry per fold.
    Use `.mean()` and `.std()` on the result to report the headline
    number and its confidence band.
    """
    pipe = make_pipeline(estimator)
    skf = _make_cv(cv)

    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
    f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")

    return acc, f1


def tune_k_cv(X, y, k_range=range(1, 31), cv: int = 5):
    """Sweep K for KNN, measure 5-fold CV accuracy for each K.

    Returns
    -------
    scores : list of (k, mean_accuracy, std_accuracy)
        One tuple per K.
    best_k : int
        The K with the highest mean accuracy.

    Why CV instead of a single train/test split for the elbow curve?
        A single-split curve is jagged and can pick a K that got lucky
        on one partition. Averaging across 5 folds smooths the curve
        and picks the K that generalizes.
    """
    skf = _make_cv(cv)
    scores = []

    for k in k_range:
        pipe = make_pipeline(KNeighborsClassifier(n_neighbors=k))
        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        scores.append((int(k), float(acc.mean()), float(acc.std())))

    # t[1] is the mean accuracy — pick the K with the highest value
    best_k = max(scores, key=lambda t: t[1])[0]
    return scores, best_k


# ======================================================================
# Algorithm comparison
# ======================================================================

def compare_algorithms(X, y, best_k: int, cv: int = 5):
    """Run three algorithms through IDENTICAL folds and report both metrics.

    Fair comparison rule:
        Same preprocessing, same splits, same scoring, same seed.
        Change any one of those and the comparison becomes apples-to-oranges.

    Parameters
    ----------
    X, y : arrays
        Full dataset (CV splits it internally).
    best_k : int
        KNN's K, taken from tune_k_cv — not guessed.

    Returns
    -------
    list of dicts, one per algorithm:
        {"name", "acc_mean", "acc_std", "f1_mean", "f1_std"}
    """
    algorithms = {
        f"KNN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = []
    for name, clf in algorithms.items():
        acc, f1 = cv_scores(clf, X, y, cv=cv)
        results.append({
            "name": name,
            "acc_mean": float(acc.mean()),
            "acc_std": float(acc.std()),
            "f1_mean": float(f1.mean()),
            "f1_std": float(f1.std()),
        })

    return results


# ======================================================================
# Final model + evaluation
# ======================================================================

def train_final(X_train, y_train, k: int) -> Pipeline:
    """Fit the final production model on the actual training split.

    The scaling is fit on X_train only — so at prediction time, the
    scaler applies the SAME transformation that was learned during fit.
    This is the correct way to prepare a model for deployment.
    """
    model = make_pipeline(KNeighborsClassifier(n_neighbors=k))
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    """Score a fitted model on the held-out test set.

    Returns
    -------
    dict with:
        accuracy          : float — fraction correct
        f1_macro          : float — unweighted mean F1 across classes
        confusion_matrix  : list[list[int]] — 3x3, rows=true, cols=pred
        predictions       : list[int] — raw predicted labels

    Why report F1 alongside accuracy?
        On balanced data (Iris is 50/50/50), they largely agree. But the
        habit matters: on imbalanced data, accuracy is misleading — a
        classifier that predicts the majority class for every input can
        hit 95% accuracy while missing every minority sample. Macro-F1
        weights every class equally and surfaces that failure.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "predictions": y_pred.tolist(),
    }