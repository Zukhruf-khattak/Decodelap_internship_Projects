"""Train/test split.

Why does this file NOT scale the data?
    Scaling must happen INSIDE each cross-validation fold, not before.
    If we scale the whole dataset first and then split, the scaler has
    already "seen" the test data — that is data leakage.

    Instead, we return raw, unscaled arrays. The scaling is handled
    automatically by the sklearn Pipeline in model.py for each fold.
"""

from sklearn.model_selection import train_test_split


def split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified shuffle-split into train/test.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Feature matrix.
    y : ndarray (n_samples,)
        Integer class labels.
    test_size : float
        Fraction reserved for testing (default 20%).
    random_state : int
        Fixed seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
        All raw, unscaled arrays.

    Why stratified?
        Iris has 50 samples per class. A random split can, by chance,
        leave a class underrepresented in the test set. `stratify=y`
        guarantees the 50/50/50 ratio is preserved in BOTH halves.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )