"""Entry point — Project 2: Data Classification Using AI (DecodeLabs).

Runs the full IPO pipeline:
    INPUT    -> load data from CSV (with sklearn fallback)
    EDA      -> print reports + save 4 plots
    PROCESS  -> 5-fold CV K-sweep + algorithm comparison
    OUTPUT   -> holdout evaluation + confusion matrix

Produces 6 PNGs in outputs/.
"""

from pathlib import Path

from src import visualize
from src.data_loader import load_data
from src.eda import print_class_stats, print_correlation, print_overview
from src.model import (
    compare_algorithms,
    evaluate,
    train_final,
    tune_k_cv,
)
from src.preprocessor import split_data


OUTPUT_DIR = Path("outputs")


def main() -> None:
    # Ensure outputs/ exists before we start saving PNGs
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Project 2: Data Classification Using AI — DecodeLabs")
    print("=" * 60 + "\n")

    # ---------------- INPUT + EDA ----------------
    X, y, feature_names, target_names = load_data()

    print_overview(X, y, feature_names, target_names)
    print_class_stats(X, y, feature_names, target_names)
    print_correlation(X, feature_names)

    print("\n[EDA] Saving plots to outputs/ ...")
    visualize.plot_class_balance(y, target_names,
                                 OUTPUT_DIR / "01_class_balance.png")
    visualize.plot_pairplot(X, y, feature_names, target_names,
                            OUTPUT_DIR / "02_pairplot.png")
    visualize.plot_correlation(X, feature_names,
                               OUTPUT_DIR / "03_correlation.png")
    visualize.plot_features_by_class(X, y, feature_names, target_names,
                                     OUTPUT_DIR / "04_features_by_class.png")
    print("  [OK] 4 EDA plots saved")

    # ---------------- PROCESS: CV-based elbow ----------------
    print("\n" + "-" * 60)
    print("Tuning K via 5-fold stratified cross-validation")
    print("-" * 60)

    k_scores, best_k = tune_k_cv(X, y, k_range=range(1, 31), cv=5)
    errors = [(1 - acc) for _, acc, _ in k_scores]
    print(f"  Best K = {best_k}  "
          f"(CV accuracy = {max(s[1] for s in k_scores):.4f})")

    visualize.plot_elbow(
        k_range=list(range(1, 31)),
        errors=errors,
        best_k=best_k,
        output_path=OUTPUT_DIR / "05_elbow_curve.png",
    )
    print("  [OK] elbow_curve.png saved")

    # ---------------- PROCESS: algorithm comparison ----------------
    print("\n" + "-" * 60)
    print("Algorithm comparison (5-fold CV)")
    print("-" * 60)

    comparison = compare_algorithms(X, y, best_k=best_k, cv=5)
    print(f"  {'Algorithm':<22} {'Acc (mean ± std)':<22} {'F1 (mean ± std)'}")
    print("  " + "-" * 62)
    for r in comparison:
        print(f"  {r['name']:<22} "
              f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}       "
              f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}")

    visualize.plot_algorithm_comparison(
        comparison,
        OUTPUT_DIR / "06_algorithm_comparison.png",
    )
    print("  [OK] algorithm_comparison.png saved")

    # ---------------- OUTPUT: final holdout evaluation ----------------
    print("\n" + "-" * 60)
    print("Final holdout evaluation (80/20 stratified split)")
    print("-" * 60)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"  Train/Test: {len(X_train)}/{len(X_test)}  "
          f"(scaling fitted inside pipeline)")

    model = train_final(X_train, y_train, k=best_k)
    results = evaluate(model, X_test, y_test)

    print(f"\n  Accuracy   : {results['accuracy']:.4f}")
    print(f"  F1 (macro) : {results['f1_macro']:.4f}")
    print("\n  Confusion matrix (rows=true, cols=pred):")
    for label, row in zip(target_names, results["confusion_matrix"]):
        print(f"    {label:<12} {row}")

    print(f"\nAll plots saved under: {OUTPUT_DIR.resolve()}")
    print("Done.\n")


if __name__ == "__main__":
    main()