# Credit Card Fraud Detection — Learning from Data

## Research question
Does the choice of validation strategy (stratified random versus chronological splitting) affect the estimated performance of machine-learning models for highly imbalanced credit-card fraud detection?

## Dataset
Download `creditcard.csv` from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the CSV at `data/creditcard.csv`. The dataset contains anonymised transaction features, `Time`, `Amount`, and binary target `Class` (1 = fraud, 0 = legitimate). It contains 284,807 transactions and 492 fraud cases in its original version.

## Run
Tested and pinned on **Python 3.13.5**.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
# Install exact pinned versions for full reproducibility:
pip install -r requirements-lock.txt
# Or minimum compatible versions:
# pip install -r requirements.txt
python analysis.py
```

## Method (Original & Revised Audited Pipeline)
- **Duplicate handling:** The primary experiment removes 1,081 exact full-row duplicates before splitting (284,807 -> 283,726 rows), consisting of 19 fraud records (492 -> 473; 3.86%) and 1,062 legitimate records (284,315 -> 283,253; 0.37%). Removing exact duplicates prevents train/test leakage across random partitions; a separate temporal sensitivity experiment runs on the raw un-deduplicated dataset (`outputs/revised/temporal_raw_sensitivity_metrics.csv`) without mixing raw and deduplicated results.
- **Chronological & random splitting:** `Time` is used strictly to order the chronological split using stable sorting (`kind="stable"`), not as a predictor. `V1`–`V28` and `Amount` are used as predictors. Both strategies partition data into 60% train, 20% validation, 20% test (`RANDOM_STATE = 42` for the primary split).
- **Models & numerical stability:** Two models are fitted inside leakage-free pipelines: class-weighted Logistic Regression and class-weighted Random Forest. In the revised pipeline, Logistic Regression uses continuous `decision_function()` log-odds scores for validation F1 threshold selection, test prediction, Average Precision, and ROC-AUC to eliminate IEEE 754 `float64` sigmoid saturation (`predict_proba == 1.0` for logits > 36.74 under `class_weight="balanced"`). Random Forest continues to use `predict_proba()[:, 1]`.
- **Prevalence & evaluation:** Every test evaluation reports test transaction count, test fraud count, test fraud prevalence, the no-skill Average Precision baseline (equal to test fraud prevalence), Average Precision (PR-AUC), ROC-AUC, precision, recall, and F1. Differences in Average Precision across splits must be interpreted alongside differing test-set fraud prevalences.
- **Repeated random split variability:** Stratified 60/20/20 random splits are repeated across seeds `42, 43, 44, 45, 46` (`outputs/revised/multi_seed_results.csv`) to quantify partition variability on the same dataset (mean and standard deviation); these five runs represent repeated partitions of one dataset, not five independent datasets.

## Outputs
- **Preserved initial outputs:** `outputs/model_metrics.csv`, `outputs/split_strategy_comparison.csv`, and initial figures in `outputs/figures/` remain untouched.
- **Confusion matrices (all 4 combinations):** `confusion_temporal_logistic_regression.png`, `confusion_temporal_random_forest.png`, `confusion_random_logistic_regression.png`, and `confusion_random_random_forest.png` are saved in `outputs/figures/` and `outputs/revised/figures/`.
- **Revised results:** Saved under `outputs/revised/` and `outputs/revised/figures/`.

## Important limitations
- The dataset covers only two days in September 2013 (~48 hours).
- `V1`–`V28` are anonymised PCA components computed prior to release, limiting substantive interpretation and introducing upstream full-dataset PCA scaling.
- Findings should not be generalised automatically to current banking systems, other regions, or longer periods.
- The chronological split tests diurnal ordering within a two-day window; it does not prove long-term concept drift.
- The dataset is highly imbalanced; accuracy alone is misleading, and test-set fraud prevalence differs between temporal and random splits.
- Thresholds are selected exclusively on validation data; the test set is never used to tune thresholds or preprocessing.

## Reproducibility and academic integrity
Run the analysis yourself. Report only values produced by your own run. Explain the code and decisions in your own words. Add citations for the dataset and any literature you actually read.
