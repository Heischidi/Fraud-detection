
"""
Learning from Data coursework: credit-card fraud detection.
Run from the project root after placing creditcard.csv in data/.
This script produces metrics and figures; it does not contain fabricated results.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve, classification_report
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the project's data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; found {list(df.columns)[:10]}...")
    if df["Class"].nunique() != 2:
        raise ValueError("Target 'Class' must contain both 0 and 1.")
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Rows loaded: {before:,}; exact duplicate rows removed: {before-len(df):,}")
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def explore(df):
    counts = df["Class"].value_counts().sort_index()
    fraud_rate = 100 * df["Class"].mean()
    print("\nDATASET SUMMARY")
    print(f"Rows after exact-deduplication: {len(df):,}")
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
    plt.savefig(FIG / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="Amount", hue="Class", bins=60, stat="density",
                 common_norm=False, element="step", log_scale=(False, True))
    plt.title("Transaction amount distribution by class")
    plt.tight_layout()
    plt.savefig(FIG / "amount_distribution.png", dpi=180)
    plt.close()

    # Plot fraud prevalence over ordered time bins, useful descriptive context.
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
    plt.savefig(FIG / "fraud_rate_over_time.png", dpi=180)
    plt.close()


def get_xy(df):
    # Time is reserved for ordering the temporal split, not used as a predictor.
    # V1-V28 are anonymised PCA components; Amount is the other original feature.
    feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["Class"].astype(int)
    return X, y, feature_cols


def split_indices(df, strategy):
    n = len(df)
    if strategy == "temporal":
        # First 60% train, next 20% validation, final 20% test.
        a, b = int(0.60*n), int(0.80*n)
        return np.arange(0, a), np.arange(a, b), np.arange(b, n)
    if strategy == "random":
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        train_val, test = train_test_split(
            idx, test_size=0.20, random_state=RANDOM_STATE,
            stratify=df["Class"]
        )
        train, val = train_test_split(
            train_val, test_size=0.25, random_state=RANDOM_STATE,
            stratify=df.iloc[train_val]["Class"]
        )
        return np.sort(train), np.sort(val), np.sort(test)
    raise ValueError("strategy must be 'temporal' or 'random'")


def build_models():
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=1500, solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])
    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=160, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])
    return {"Logistic Regression": logistic, "Random Forest": forest}


def best_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "average_precision_PR_AUC": average_precision_score(y_true, probabilities),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "threshold": threshold,
        "fraud_cases_in_test": int(np.sum(y_true)),
        "test_transactions": int(len(y_true)),
    }, pred


def run_strategy(df, X, y, strategy, save_confusion=True):
    train_idx, val_idx, test_idx = split_indices(df, strategy)
    print(f"\n{strategy.upper()} SPLIT")
    for name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        print(f"{name}: n={len(idx):,}; frauds={int(y.iloc[idx].sum()):,}")

    results = []
    fitted = {}
    for name, model in build_models().items():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        threshold = best_f1_threshold(y.iloc[val_idx], val_prob)
        test_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        metrics, pred = evaluate(y.iloc[test_idx], test_prob, threshold)
        metrics.update({"strategy": strategy, "model": name})
        results.append(metrics)
        fitted[name] = (model, test_prob, pred, y.iloc[test_idx].to_numpy())
        print(f"\n{name}: {metrics}")
        print(classification_report(y.iloc[test_idx], pred, digits=4, zero_division=0))

        if save_confusion and strategy == "temporal":
            cm = confusion_matrix(y.iloc[test_idx], pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Legitimate", "Fraud"],
                        yticklabels=["Legitimate", "Fraud"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Temporal test confusion matrix: {name}")
            plt.tight_layout()
            safe = name.lower().replace(" ", "_")
            plt.savefig(FIG / f"confusion_{safe}.png", dpi=180)
            plt.close()
    return results, fitted


def main():
    df = load_data()
    explore(df)
    X, y, features = get_xy(df)
    print(f"\nPredictor count: {len(features)}")
    print("Predictors:", features)

    # Main analysis: chronological train/validation/test.
    temporal_results, temporal_models = run_strategy(df, X, y, "temporal")

    # Student-designed experiment: same algorithms and predictors, different split strategy.
    random_results, _ = run_strategy(df, X, y, "random", save_confusion=False)

    all_results = pd.DataFrame(temporal_results + random_results)
    all_results.to_csv(OUT / "model_metrics.csv", index=False)

    pivot = all_results.pivot(index="model", columns="strategy",
                              values="average_precision_PR_AUC")
    pivot.to_csv(OUT / "split_strategy_comparison.csv")

    plt.figure(figsize=(8, 4))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Average precision (PR-AUC)")
    plt.title("Effect of validation strategy on estimated test performance")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "split_strategy_comparison.png", dpi=180)
    plt.close()

    print("\nFINAL METRICS")
    print(all_results.to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")
    print("IMPORTANT: Interpret the actual metrics and fraud counts; do not assume one split is better.")


if __name__ == "__main__":
    main()
