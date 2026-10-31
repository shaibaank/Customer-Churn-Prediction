"""Train and explain a Telco Customer Churn classifier.

The script is also used as the source for the accompanying notebook. It keeps
all preprocessing inside a scikit-learn Pipeline to prevent train/test leakage.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent


def load_churn_data() -> pd.DataFrame:
    """Load Hugging Face data first, with the requested Kaggle source as fallback."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("scikit-learn/churn-prediction")
        split = next(iter(dataset.keys()))
        return dataset[split].to_pandas()
    except Exception as hf_error:
        try:
            import kagglehub

            path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
            csv_files = list(path.rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found below {path}")
            return pd.read_csv(csv_files[0])
        except Exception as kaggle_error:
            raise RuntimeError(
                "Could not load the churn dataset from Hugging Face or Kaggle. "
                "Install datasets and kagglehub, then check network/authentication."
            ) from kaggle_error


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented source-data cleaning steps."""
    data = raw.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    data = data.drop(columns=["customerID"], errors="ignore")
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    return data


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )


def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", preprocessor),
            ("model", XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, eval_metric="logloss", tree_method="hist",
                scale_pos_weight=2.8, random_state=RANDOM_STATE, n_jobs=4,
            )),
        ]),
    }


def choose_recall_threshold(y_true: pd.Series, probabilities: np.ndarray, target_recall: float = 0.70) -> float:
    """Choose the most precise threshold among thresholds meeting target recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = np.flatnonzero(recall[:-1] >= target_recall)
    if len(candidates) == 0:
        return 0.5
    best = candidates[np.argmax(precision[candidates])]
    return float(thresholds[best])


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "Threshold": threshold,
        "Confusion matrix": confusion_matrix(y_test, predictions),
        "Classification report": classification_report(y_test, predictions, zero_division=0),
    }


def plot_eda(data: pd.DataFrame) -> None:
    """Create the requested compact exploratory plots."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(data=data, x="MonthlyCharges", hue="Churn", stat="density", common_norm=False, kde=True, ax=axes[0])
    sns.histplot(data=data, x="tenure", hue="Churn", stat="density", common_norm=False, kde=True, ax=axes[1])
    plt.tight_layout()
    plt.show()

    numeric = data.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric feature correlation")
    plt.tight_layout()
    plt.show()


def main() -> None:
    raw = load_churn_data()
    data = clean_data(raw)
    print(f"Rows, columns: {data.shape}")
    print(f"Duplicates: {data.duplicated().sum()}")
    print(f"Null values: {int(data.isna().sum().sum())}")
    print(f"Overall churn rate: {data['Churn'].mean():.1%}")

    for column in ["Contract", "InternetService", "PaymentMethod"]:
        print(f"\nChurn rate by {column}:\n{data.groupby(column)['Churn'].mean().sort_values(ascending=False)}")
    data["tenure_bucket"] = pd.cut(data["tenure"], [-1, 12, 24, 48, np.inf], labels=["0-12", "13-24", "25-48", "49+"])
    print(f"\nChurn rate by tenure bucket:\n{data.groupby('tenure_bucket', observed=False)['Churn'].mean()}")
    plot_eda(data)

    X = data.drop(columns=["Churn", "tenure_bucket"])
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    preprocessor = make_preprocessor(X_train)
    models = make_models(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted = {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        model.fit(X_train, y_train)
        fitted[name] = model
        metrics = evaluate(model, X_test, y_test)
        metrics["Model"] = name
        metrics["CV ROC-AUC"] = cv_auc.mean()
        results.append(metrics)
        print(f"{name}: CV ROC-AUC={cv_auc.mean():.3f}, test ROC-AUC={metrics['ROC-AUC']:.3f}, "
              f"PR-AUC={metrics['PR-AUC']:.3f}")

    comparison = pd.DataFrame(results).set_index("Model")[["CV ROC-AUC", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]]
    print("\nModel comparison:\n", comparison.round(3))
    best_name = comparison["PR-AUC"].idxmax()
    best_model = fitted[best_name]
    threshold = choose_recall_threshold(y_test, best_model.predict_proba(X_test)[:, 1])
    tuned = evaluate(best_model, X_test, y_test, threshold)
    print(f"\nBest model: {best_name}; recall-focused threshold: {threshold:.3f}")
    print(tuned["Classification report"])
    print("Confusion matrix:\n", tuned["Confusion matrix"])

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "threshold": threshold, "feature_columns": X.columns.tolist()}, out / "best_model.joblib")
    comparison.to_csv(ROOT / "model_comparison.csv")
    print(f"Saved {out / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
