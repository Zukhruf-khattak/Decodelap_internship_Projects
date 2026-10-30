"""End-to-end tests for the classification pipeline.

Grouped by IPO phase so failures immediately tell you where to look.
"""

import numpy as np
import pytest

from src.data_loader import load_data
from src.model import (
    compare_algorithms,
    cv_scores,
    evaluate,
    make_pipeline,
    train_final,
    tune_k_cv,
)
from src.preprocessor import split_data


# ======================================================================
# INPUT
# ======================================================================

def test_dataset_shape():
    """Iris should always be 150 x 4 with 3 classes."""
    X, y, features, targets = load_data()
    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert len(features) == 4
    assert len(targets) == 3


def test_dataset_is_balanced():
    """50 samples per class — any deviation means the CSV is corrupt."""
    _, y, _, _ = load_data()
    assert list(np.bincount(y)) == [50, 50, 50]


# ======================================================================
# PREPROCESS
# ======================================================================

def test_split_sizes():
    """80/20 split on 150 samples -> 120 train / 30 test."""
    X, y, _, _ = load_data()
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2)
    assert len(X_tr) == 120
    assert len(X_te) == 30


def test_split_preserves_class_balance():
    """Stratified split must contain all 3 classes in both halves."""
    X, y, _, _ = load_data()
    _, _, y_tr, y_te = split_data(X, y, test_size=0.2)
    assert set(np.unique(y_tr)) == {0, 1, 2}
    assert set(np.unique(y_te)) == {0, 1, 2}


def test_pipeline_scales_features():
    """StandardScaler inside Pipeline should produce mean~0, std~1."""
    from sklearn.neighbors import KNeighborsClassifier
    X, y, _, _ = load_data()
    pipe = make_pipeline(KNeighborsClassifier(n_neighbors=5))
    pipe.fit(X, y)
    scaled = pipe.named_steps["scaler"].transform(X)
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-8)


# ======================================================================
# CV / TUNING
# ======================================================================

def test_cv_scores_return_five_folds():
    """Each scorer must produce exactly 5 values (one per fold)."""
    from sklearn.neighbors import KNeighborsClassifier
    X, y, _, _ = load_data()
    acc, f1 = cv_scores(KNeighborsClassifier(n_neighbors=5), X, y, cv=5)
    assert len(acc) == 5
    assert len(f1) == 5
    assert 0.0 <= acc.mean() <= 1.0


def test_tune_k_returns_valid_best_k():
    """Sweeping K=1..10 must yield a best_k in that range."""
    X, y, _, _ = load_data()
    scores, best_k = tune_k_cv(X, y, k_range=range(1, 11), cv=5)
    assert len(scores) == 10
    assert 1 <= best_k <= 10


# ======================================================================
# COMPARISON
# ======================================================================

def test_compare_algorithms_returns_three_rows():
    """Must compare exactly 3 algorithms: KNN, Decision Tree, LogReg."""
    X, y, _, _ = load_data()
    results = compare_algorithms(X, y, best_k=5, cv=5)
    assert len(results) == 3
    names = [r["name"] for r in results]
    assert any("KNN" in n for n in names)
    assert any("Decision Tree" in n for n in names)
    assert any("Logistic" in n for n in names)


def test_comparison_values_are_reasonable():
    """No algorithm should score below 0.80 on Iris."""
    X, y, _, _ = load_data()
    results = compare_algorithms(X, y, best_k=5, cv=5)
    for r in results:
        assert 0.80 <= r["acc_mean"] <= 1.00
        assert 0.80 <= r["f1_mean"] <= 1.00


# ======================================================================
# FINAL EVAL
# ======================================================================

def test_final_model_high_accuracy():
    """KNN with K=5 should hit at least 90% on holdout."""
    X, y, _, _ = load_data()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = train_final(X_tr, y_tr, k=5)
    results = evaluate(model, X_te, y_te)
    assert results["accuracy"] >= 0.90
    assert results["f1_macro"] >= 0.90


def test_confusion_matrix_shape():
    """3-class problem -> 3x3 confusion matrix."""
    X, y, _, _ = load_data()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    model = train_final(X_tr, y_tr, k=5)
    results = evaluate(model, X_te, y_te)
    cm = results["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)