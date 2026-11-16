"""
Learning from Data coursework: credit-card fraud detection.
Run from the project root after placing creditcard.csv in data/.
This script preserves original baseline outputs while running the revised
methodologically audited pipeline and writing revised results to outputs/revised/.
"""
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REV_OUT = OUT / "revised"
REV_FIG = REV_OUT / "figures"

OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
REV_OUT.mkdir(parents=True, exist_ok=True)
REV_FIG.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MULTI_SEEDS = [42, 43, 44, 45, 46]


def load_data(deduplicate=True, save_summary=False):
    """Load creditcard.csv, validate schema, report duplicates, and sort stably by Time."""
    if not DATA_PATH.exists() or not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the project's data/ folder."
        )
    try:
        df = pd.read_csv(DATA_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise RuntimeError(f"Failed to read dataset from {DATA_PATH}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset is empty: {DATA_PATH}")

    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; found {list(df.columns)[:10]}...")

    if df["Class"].isna().any():
        raise ValueError("Target 'Class' contains missing values.")
    if not set(df["Class"].unique()).issubset({0, 1}) or df["Class"].nunique() != 2:
        raise ValueError("Target 'Class' must contain only binary values 0 and 1.")

    raw_total = len(df)
    raw_fraud = int((df["Class"] == 1).sum())
    raw_legit = int((df["Class"] == 0).sum())

    dup_mask = df.duplicated(keep="first")
    dup_total = int(dup_mask.sum())
    dup_fraud = int((df.loc[dup_mask, "Class"] == 1).sum())
    dup_legit = int((df.loc[dup_mask, "Class"] == 0).sum())

    if deduplicate:
        df = df.loc[~dup_mask].copy()

    dedup_total = len(df)
    dedup_fraud = int((df["Class"] == 1).sum())
    dedup_legit = int((df["Class"] == 0).sum())

    label = "DEDUPLICATED DATASET" if deduplicate else "RAW DATASET (NO DEDUPLICATION)"
    print(f"\n{label}")
    print(
        f"Original rows: {raw_total:,} (fraud={raw_fraud:,}, legitimate={raw_legit:,}, "
        f"prevalence={100.0 * raw_fraud / raw_total:.4f}%)"
    )
    print(
        f"Exact full-row duplicates identified: {dup_total:,} "
        f"(fraud={dup_fraud:,} [{100.0 * dup_fraud / raw_fraud:.2f}%], "
        f"legitimate={dup_legit:,} [{100.0 * dup_legit / raw_legit:.2f}%])"
    )
    if deduplicate:
        print(
            f"Rows retained after exact deduplication: {dedup_total:,} "
            f"(fraud={dedup_fraud:,}, legitimate={dedup_legit:,}, "
            f"prevalence={100.0 * dedup_fraud / dedup_total:.4f}%)"
        )
        print(
            "Note: Removing 19 repeated fraud rows prevents exact-row train/test leakage in "
            "random splits, though it disproportionately reduces fraud cases (3.86% vs 0.37% "
            "for legitimate) and may discard legitimate same-second burst fraud repeats."
        )
    else:
        print("Rows retained (deduplication disabled for temporal sensitivity check):", f"{dedup_total:,}")

    if save_summary and deduplicate:
        summary_rows = [
            {
                "category": "Total transactions",
                "original_count": raw_total,
                "duplicates_removed": dup_total,
                "deduplicated_count": dedup_total,
                "pct_removed": 100.0 * dup_total / raw_total,
            },
            {
                "category": "Fraud (Class=1)",
                "original_count": raw_fraud,
                "duplicates_removed": dup_fraud,
                "deduplicated_count": dedup_fraud,
                "pct_removed": 100.0 * dup_fraud / raw_fraud,
            },
            {
                "category": "Legitimate (Class=0)",
                "original_count": raw_legit,
                "duplicates_removed": dup_legit,
                "deduplicated_count": dedup_legit,
                "pct_removed": 100.0 * dup_legit / raw_legit,
            },
        ]
        pd.DataFrame(summary_rows).to_csv(REV_OUT / "deduplication_summary.csv", index=False)

    # Stable sort guarantees deterministic tie-ordering for identical Time timestamps.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return df


def explore(df, fig_dir=REV_FIG):
    """Print dataset summary statistics and save exploratory plots to fig_dir."""
    counts = df["Class"].value_counts().sort_index()
    fraud_rate = 100.0 * df["Class"].mean()
    print("\nDATASET SUMMARY")
    print(f"Rows after preprocessing: {len(df):,}")
    print(f"Fraud cases: {int(counts.get(1, 0)):,}")
    print(f"Legitimate cases: {int(counts.get(0, 0)):,}")
    print(f"Fraud prevalence: {fraud_rate:.4f}%")
    print(f"Missing cells: {int(df.isna().sum().sum()):,}")
    print("\nClass counts:\n", counts)
    print("\nDescriptive statistics:\n", df[["Time", "Amount"]].describe())

    plt.figure(figsize=(7, 4))
    sns.barplot(x=["Legitimate", "Fraud"], y=[counts.get(0, 0), counts.get(1, 0)])
    plt.yscale("log")
    plt.ylabel("Number of transactions (log scale)")
    plt.title("Class distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        log_scale=(False, True),
    )
    plt.title("Transaction amount distribution by class")
    plt.tight_layout()
    plt.savefig(fig_dir / "amount_distribution.png", dpi=180)
    plt.close()

    temp = df[["Time", "Class"]].copy()
    temp["time_bin"] = pd.qcut(temp["Time"], q=20, duplicates="drop")
    by_time = temp.groupby("time_bin", observed=True)["Class"].mean().reset_index()
    by_time["bin"] = np.arange(1, len(by_time) + 1)
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=by_time, x="bin", y="Class", marker="o")
    plt.xlabel("Chronological time bin (earliest to latest)")
    plt.ylabel("Fraud proportion")
    plt.title("Fraud prevalence across chronological time bins")
    plt.tight_layout()
    plt.savefig(fig_dir / "fraud_rate_over_time.png", dpi=180)
    plt.close()


def get_xy(df):
    """Extract V1-V28 and Amount predictors while excluding Time and Class."""
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    if len(feature_cols) <= 1:
        raise ValueError(f"Expected PCA features V1..V28 plus Amount; found {feature_cols}")
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["Class"].astype(int)
    return X, y, feature_cols


def split_indices(df, strategy, seed=RANDOM_STATE):
    """Partition row indices into 60% train, 20% validation, 20% test."""
    n = len(df)
    if n < 10:
        raise ValueError(f"Dataset too small to split (n={n}).")

    if strategy == "temporal":
        a, b = int(0.60 * n), int(0.80 * n)
        train, val, test = np.arange(0, a), np.arange(a, b), np.arange(b, n)
    elif strategy == "random":
        idx = np.arange(n)
        train_val, test = train_test_split(
            idx, test_size=0.20, random_state=seed, stratify=df["Class"]
        )
        train, val = train_test_split(
            train_val, test_size=0.25, random_state=seed, stratify=df.iloc[train_val]["Class"]
        )
        train, val, test = np.sort(train), np.sort(val), np.sort(test)
    else:
        raise ValueError("strategy must be 'temporal' or 'random'")

    for split_name, split_idx in [("train", train), ("validation", val), ("test", test)]:
        if len(split_idx) == 0:
            raise ValueError(f"{strategy} {split_name} split is empty.")
        if df["Class"].iloc[split_idx].nunique() < 2:
            raise ValueError(f"{strategy} {split_name} split does not contain both classes.")

    return train, val, test


def build_models(seed=RANDOM_STATE):
    """Build class-weighted Logistic Regression and Random Forest pipelines."""
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            solver="lbfgs",
            random_state=seed,
        )),
    ])
    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )),
    ])
    return {"Logistic Regression": logistic, "Random Forest": forest}


def get_continuous_scores(model, X):
    """Return continuous decision scores for LR (avoiding float64 saturation) and predict_proba for RF."""
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        score_type = "decision_function"
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
        score_type = "predict_proba"
    else:
        raise TypeError(f"Model {type(model).__name__} exposes neither decision_function nor predict_proba.")

    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite scores returned by {type(model).__name__} ({score_type}).")
    return scores, score_type


def best_f1_threshold(y_true, scores):
    """Select decision threshold maximizing F1 strictly on validation labels and scores."""
    y_arr = np.asarray(y_true)
    s_arr = np.asarray(scores, dtype=np.float64)
    if len(y_arr) == 0 or len(y_arr) != len(s_arr):
        raise ValueError("y_true and scores must be non-empty and of equal length.")
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("Scores passed to best_f1_threshold contain non-finite values.")

    precision, recall, thresholds = precision_recall_curve(y_arr, s_arr)
    if len(thresholds) == 0:
        return 0.0 if np.min(s_arr) < 0 else 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if len(f1) == 0 or np.all(np.isnan(f1)):
        raise ValueError("Unable to compute F1 across precision-recall thresholds.")
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y_true, scores, threshold, score_type):
    """Compute ranking and thresholded metrics along with test-set prevalence baselines."""
    y_arr = np.asarray(y_true)
    s_arr = np.asarray(scores, dtype=np.float64)
    if len(y_arr) == 0 or len(y_arr) != len(s_arr):
        raise ValueError("y_true and scores must be non-empty and of equal length.")
    if len(np.unique(y_arr)) < 2:
        raise ValueError("Evaluation target must contain both positive and negative classes.")

    pred = (s_arr >= threshold).astype(int)
    n_test = int(len(y_arr))
    n_fraud = int(np.sum(y_arr))
    prevalence = float(n_fraud / n_test)

    return {
        "average_precision_PR_AUC": float(average_precision_score(y_arr, s_arr)),
        "no_skill_ap_baseline": prevalence,
        "ROC_AUC": float(roc_auc_score(y_arr, s_arr)),
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_arr, pred, zero_division=0)),
        "F1": float(f1_score(y_arr, pred, zero_division=0)),
        "threshold": float(threshold),
        "score_type": score_type,
        "fraud_cases_in_test": n_fraud,
        "test_transactions": n_test,
        "test_fraud_prevalence": prevalence,
        "test_fraud_prevalence_pct": float(100.0 * prevalence),
    }, pred


def save_confusion_matrix_figure(y_true, pred, strategy, model_name):
    """Render confusion matrix with strategy-specific filename to both outputs/revised/figures and outputs/figures."""
    cm = confusion_matrix(y_true, pred)
    safe = model_name.lower().replace(" ", "_")
    filename = f"confusion_{strategy}_{safe}.png"

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{strategy.capitalize()} test confusion matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(REV_FIG / filename, dpi=180)
    plt.savefig(FIG / filename, dpi=180)
    plt.close()


def run_strategy(
    df,
    X,
    y,
    strategy,
    seed=RANDOM_STATE,
    save_confusion=True,
    dataset_variant="deduplicated",
    collect_lr_saturation=False,
    verbose=True,
):
    """Fit models on train split, tune threshold on validation split, and evaluate once on test split."""
    train_idx, val_idx, test_idx = split_indices(df, strategy, seed=seed)
    if verbose:
        print(f"\n{strategy.upper()} SPLIT ({dataset_variant}, seed={seed})")
        for split_name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
            n_split = len(idx)
            n_frauds = int(y.iloc[idx].sum())
            print(
                f"{split_name}: n={n_split:,}; frauds={n_frauds:,}; "
                f"prevalence={100.0 * n_frauds / n_split:.4f}%"
            )

    results = []
    fitted = {}
    lr_saturation_rows = []

    for name, model in build_models(seed=RANDOM_STATE).items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        val_scores, score_type = get_continuous_scores(model, X.iloc[val_idx])
        threshold = best_f1_threshold(y.iloc[val_idx], val_scores)

        test_scores, _ = get_continuous_scores(model, X.iloc[test_idx])
        metrics, pred = evaluate(y.iloc[test_idx], test_scores, threshold, score_type)
        metrics.update({
            "strategy": strategy,
            "model": name,
            "seed": int(seed),
            "dataset_variant": dataset_variant,
        })
        results.append(metrics)
        fitted[name] = (model, test_scores, pred, y.iloc[test_idx].to_numpy())

        if verbose:
            print(f"\n{name} ({score_type}): {metrics}")
            print(classification_report(y.iloc[test_idx], pred, digits=4, zero_division=0))

        if save_confusion:
            save_confusion_matrix_figure(y.iloc[test_idx], pred, strategy, name)

        if collect_lr_saturation and name == "Logistic Regression":
            val_prob = np.asarray(model.predict_proba(X.iloc[val_idx])[:, 1], dtype=np.float64)
            test_prob = np.asarray(model.predict_proba(X.iloc[test_idx])[:, 1], dtype=np.float64)
            th_prob = best_f1_threshold(y.iloc[val_idx], val_prob)
            prob_metrics, _ = evaluate(y.iloc[test_idx], test_prob, th_prob, "predict_proba")

            y_val_arr = y.iloc[val_idx].to_numpy()
            y_test_arr = y.iloc[test_idx].to_numpy()
            val_sat_total = int(np.sum(val_prob == 1.0))
            val_sat_fraud = int(np.sum((val_prob == 1.0) & (y_val_arr == 1)))
            test_sat_total = int(np.sum(test_prob == 1.0))
            test_sat_fraud = int(np.sum((test_prob == 1.0) & (y_test_arr == 1)))

            lr_saturation_rows.append({
                "strategy": strategy,
                "evaluation_mode": "original_predict_proba",
                "score_type": "predict_proba",
                "threshold": prob_metrics["threshold"],
                "average_precision_PR_AUC": prob_metrics["average_precision_PR_AUC"],
                "ROC_AUC": prob_metrics["ROC_AUC"],
                "precision": prob_metrics["precision"],
                "recall": prob_metrics["recall"],
                "F1": prob_metrics["F1"],
                "val_prob_eq_1_count": val_sat_total,
                "val_fraud_prob_eq_1_count": val_sat_fraud,
                "test_prob_eq_1_count": test_sat_total,
                "test_fraud_prob_eq_1_count": test_sat_fraud,
                "val_max_logit": float(np.max(val_scores)),
                "test_max_logit": float(np.max(test_scores)),
            })
            lr_saturation_rows.append({
                "strategy": strategy,
                "evaluation_mode": "revised_decision_function",
                "score_type": "decision_function",
                "threshold": metrics["threshold"],
                "average_precision_PR_AUC": metrics["average_precision_PR_AUC"],
                "ROC_AUC": metrics["ROC_AUC"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "F1": metrics["F1"],
                "val_prob_eq_1_count": val_sat_total,
                "val_fraud_prob_eq_1_count": val_sat_fraud,
                "test_prob_eq_1_count": test_sat_total,
                "test_fraud_prob_eq_1_count": test_sat_fraud,
                "val_max_logit": float(np.max(val_scores)),
                "test_max_logit": float(np.max(test_scores)),
            })

    return results, fitted, lr_saturation_rows


def run_multi_seed_random(df, X, y, seeds, cached_seed42_results=None):
    """Run stratified 60/20/20 random splits across multiple seeds to measure partition variance."""
    per_seed_rows = []
    for seed in seeds:
        if cached_seed42_results is not None and seed == RANDOM_STATE:
            seed_results = [dict(row) for row in cached_seed42_results]
        else:
            seed_results, _, _ = run_strategy(
                df,
                X,
                y,
                "random",
                seed=seed,
                save_confusion=False,
                dataset_variant="deduplicated",
                collect_lr_saturation=False,
                verbose=False,
            )
        for row in seed_results:
            row["row_type"] = "per_seed"
            per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    metric_cols = ["average_precision_PR_AUC", "ROC_AUC", "precision", "recall", "F1"]

    summary_records = []
    combined_rows = list(per_seed_rows)
    for model_name, group in per_seed_df.groupby("model", sort=False):
        mean_row = {
            "row_type": "summary_mean",
            "strategy": "random",
            "model": model_name,
            "seed": "mean_5_seeds",
            "dataset_variant": "deduplicated",
            "score_type": group["score_type"].iloc[0],
            "fraud_cases_in_test": float(group["fraud_cases_in_test"].mean()),
            "test_transactions": float(group["test_transactions"].mean()),
            "test_fraud_prevalence": float(group["test_fraud_prevalence"].mean()),
            "test_fraud_prevalence_pct": float(group["test_fraud_prevalence_pct"].mean()),
            "no_skill_ap_baseline": float(group["no_skill_ap_baseline"].mean()),
            "threshold": float(group["threshold"].mean()),
        }
        std_row = {
            "row_type": "summary_std",
            "strategy": "random",
            "model": model_name,
            "seed": "std_ddof1_5_seeds",
            "dataset_variant": "deduplicated",
            "score_type": group["score_type"].iloc[0],
            "fraud_cases_in_test": float(group["fraud_cases_in_test"].std(ddof=1)),
            "test_transactions": float(group["test_transactions"].std(ddof=1)),
            "test_fraud_prevalence": float(group["test_fraud_prevalence"].std(ddof=1)),
            "test_fraud_prevalence_pct": float(group["test_fraud_prevalence_pct"].std(ddof=1)),
            "no_skill_ap_baseline": float(group["no_skill_ap_baseline"].std(ddof=1)),
            "threshold": float(group["threshold"].std(ddof=1)),
        }
        for col in metric_cols:
            mean_val = float(group[col].mean())
            std_val = float(group[col].std(ddof=1))
            mean_row[col] = mean_val
            std_row[col] = std_val
            summary_records.append({
                "model": model_name,
                "metric": col,
                "n_seeds": len(seeds),
                "mean": mean_val,
                "std_ddof1": std_val,
                "min": float(group[col].min()),
                "max": float(group[col].max()),
            })
        combined_rows.extend([mean_row, std_row])

    multi_seed_df = pd.DataFrame(combined_rows)
    summary_df = pd.DataFrame(summary_records)
    multi_seed_df.to_csv(REV_OUT / "multi_seed_results.csv", index=False)
    summary_df.to_csv(REV_OUT / "multi_seed_summary.csv", index=False)
    return multi_seed_df, summary_df


def main():
    print(f"Python version: {sys.version.split()[0]}")
    df = load_data(deduplicate=True, save_summary=True)
    explore(df, fig_dir=REV_FIG)
    X, y, features = get_xy(df)
    print(f"\nPredictor count: {len(features)}")
    print("Predictors:", features)

    # 1. Main revised analysis on deduplicated data (seed=42 for random split)
    temporal_results, _, temporal_lr_sat = run_strategy(
        df,
        X,
        y,
        "temporal",
        seed=RANDOM_STATE,
        save_confusion=True,
        dataset_variant="deduplicated",
        collect_lr_saturation=True,
    )
    random_results, _, random_lr_sat = run_strategy(
        df,
        X,
        y,
        "random",
        seed=RANDOM_STATE,
        save_confusion=True,
        dataset_variant="deduplicated",
        collect_lr_saturation=True,
    )

    revised_results = pd.DataFrame(temporal_results + random_results)
    revised_results.to_csv(REV_OUT / "model_metrics_revised.csv", index=False)

    lr_sat_df = pd.DataFrame(temporal_lr_sat + random_lr_sat)
    lr_sat_df.to_csv(REV_OUT / "lr_saturation_comparison.csv", index=False)

    pivot = revised_results.pivot(
        index="model", columns="strategy", values="average_precision_PR_AUC"
    )
    pivot.to_csv(REV_OUT / "split_strategy_comparison_revised.csv")

    plt.figure(figsize=(8, 4))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Average Precision (AP)")
    plt.title("Revised split strategy comparison (continuous LR decision scores)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REV_FIG / "split_strategy_comparison_revised.png", dpi=180)
    plt.close()

    # 2. Separate temporal sensitivity experiment on raw data without deduplication
    df_raw = load_data(deduplicate=False, save_summary=False)
    X_raw, y_raw, _ = get_xy(df_raw)
    raw_temporal_results, _, _ = run_strategy(
        df_raw,
        X_raw,
        y_raw,
        "temporal",
        seed=RANDOM_STATE,
        save_confusion=False,
        dataset_variant="raw_no_deduplication",
        collect_lr_saturation=False,
    )
    raw_sensitivity_df = pd.DataFrame(raw_temporal_results)
    raw_sensitivity_df.to_csv(REV_OUT / "temporal_raw_sensitivity_metrics.csv", index=False)

    # 3. Repeated 60/20/20 stratified random split across 5 seeds (same dataset, partition variance)
    multi_seed_df, multi_seed_summary_df = run_multi_seed_random(
        df, X, y, MULTI_SEEDS, cached_seed42_results=random_results
    )

    print("\nREVISED FINAL METRICS (DEDUPLICATED)")
    print(revised_results.to_string(index=False))
    print("\nLOGISTIC REGRESSION SATURATION COMPARISON")
    print(lr_sat_df.to_string(index=False))
    print("\nTEMPORAL RAW SENSITIVITY METRICS (NO DEDUPLICATION)")
    print(raw_sensitivity_df.to_string(index=False))
    print("\nMULTI-SEED RANDOM SPLIT SUMMARY (5 SEEDS ON SAME DATASET)")
    print(multi_seed_summary_df.to_string(index=False))
    print(f"\nSaved revised outputs to: {REV_OUT}")
    print(f"Original baseline files preserved in: {OUT}")


if __name__ == "__main__":
    main()
