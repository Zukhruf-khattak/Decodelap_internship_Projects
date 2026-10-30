# Project 2 — Data Classification Using AI

**Author:** AI Engineering Intern (Zukhruf)
**Organization:** DecodeLabs
**Batch:** 2026
**Track:** Artificial Intelligence — Predictive Phase

---

## 1. Executive Summary
Project 1 introduced deterministic logic (hand-written rules). Project 2 transitions into the predictive phase: **Supervised Learning**. Instead of writing the rules, we provide the machine with labeled history (the Iris dataset) and allow it to derive the decision boundary. This report details the full pipeline: Exploratory Data Analysis (EDA), data preprocessing, model tuning via Cross-Validation, and rigorous evaluation using multiple metrics and algorithm comparisons.

## 2. Dataset Overview
The **Iris Benchmark** is used for this classification task.
- **Samples:** 150 (perfectly balanced: 50 per class)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Dimensions (Features):** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width in cm)
- **Data Quality:** Clean. Zero missing values.
- **EDA Findings:** Petal length and petal width are highly correlated (r > 0.96). Setosa is linearly separable from the other two classes, while Versicolor and Virginica have slight overlap.

## 3. Methodology — The IPO Framework
The pipeline strictly follows the Input-Process-Output architecture:

- **INPUT:** Load the dataset from `data/iris.csv` (with a fallback to scikit-learn's built-in dataset).
- **EDA:** Print statistical summaries and generate 4 visual plots to understand feature distributions and correlations.
- **PROCESS:** 
  - Split data into 80% Train / 20% Test (Stratified to preserve class ratios).
  - Apply `StandardScaler` (Mean=0, Variance=1) *inside* the CV pipeline to prevent data leakage.
  - Tune the K-Nearest Neighbors (KNN) algorithm using 5-Fold Cross-Validation.
- **OUTPUT:** Evaluate the final model using Accuracy, Macro-F1 Score, and a Confusion Matrix. Compare KNN against Decision Tree and Logistic Regression.

## 4. Algorithm & Implementation Details
### 4.1 Why K-Nearest Neighbors (KNN)?
KNN is a proximity-based algorithm: "Similar things exist in close proximity." It classifies a new data point based on the majority vote of its `K` nearest neighbors in the feature space.

### 4.2 The Gatekeeper Rule: Scaling
Raw features have different scales. Distance-based algorithms like KNN will let larger-magnitude features dominate the distance calculation. We use `StandardScaler` to ensure all features contribute equally. 
*Crucially, scaling is fitted ONLY on the training folds to prevent data leakage.*

### 4.3 Tuning the Engine: Choosing "K"
We sweep `K` from 1 to 30 using 5-fold stratified cross-validation. 
- `K=1`: Overfitting (captures noise).
- `K=100`: Underfitting (too generic).
- **The Elbow:** We select the `K` that minimizes the CV error rate.

### 4.4 Data Leakage Prevention
All models are wrapped in a Scikit-Learn `Pipeline([StandardScaler, Classifier])`. During CV, the scaler refits on the training fold and transforms the held-out fold. Naively scaling the whole dataset before CV would leak future information and artificially inflate scores by 1–3%.

## 5. Evaluation Metrics
Accuracy alone is insufficient. We use:
- **Accuracy:** Overall correctness.
- **Macro-F1:** The harmonic mean of Precision and Recall, averaged across all 3 classes. It ensures we aren't ignoring a minority class (though Iris is balanced, this is a best practice).
- **Confusion Matrix:** A 3x3 grid detailing exactly where the model makes mistakes (False Positives vs False Negatives).

## 6. Results & Algorithm Comparison
We compared three algorithms on identical folds and preprocessing steps:

| Algorithm | CV Accuracy (Mean ± Std) | CV Macro-F1 |
| :--- | :--- | :--- |
| **KNN (Tuned K)** | ~0.96 ± 0.02 | ~0.96 |
| **Logistic Regression** | ~0.95 ± 0.02 | ~0.95 |
| **Decision Tree** | ~0.94 ± 0.03 | ~0.94 |

**Final Holdout Evaluation:**
- **Accuracy:** ~93% – 97% (depending on the split seed)
- **Confusion Matrix:** Setosa is perfectly separated. The only minor confusion is between Versicolor and Virginica (1 or 2 misclassifications).

## 7. Artifacts Generated
Running `main.py` generates the following visual artifacts in the `outputs/` folder:
1. `01_class_balance.png` — Confirms 50/50/50 distribution.
2. `02_pairplot.png` — Visualizes feature interactions by species.
3. `03_correlation.png` — Heatmap of Pearson correlation.
4. `04_features_by_class.png` — Boxplots showing feature distributions.
5. `05_elbow_curve.png` — Error rate vs. K value, highlighting the optimal K.
6. `06_algorithm_comparison.png` — Bar chart comparing KNN, DT, and LogReg.

## 8. Lessons Learned
1. **Scaling is mandatory** for distance-based models.
2. **Cross-Validation is cheap**, but single-split variance is expensive in terms of trust.
3. **Compare algorithms on identical folds**; anything else is an unfair comparison.
4. **Accuracy is a mirage** on imbalanced data. Always pair it with F1 and a Confusion Matrix.

## 9. Conclusion
Project 2 successfully bridges the gap between deterministic logic and probabilistic learning. By mastering the full pipeline (EDA → Split → Scale → CV Tune → Compare → Evaluate), we have built the exact same skeleton used in production systems for models 1000x larger. The foundation is now set for Deep Learning and CNNs in Project 3.