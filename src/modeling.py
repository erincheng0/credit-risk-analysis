import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def train_val_test_split(df, target, test_size=0.2, val_size=0.25, random_state=42):
    x = df.drop(columns=[target])
    y = df[target]

    # 1) Train+val vs test
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2) Train vs val (IMPORTANT: use x_trainval and y_trainval here)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test



def scale_numeric(
        x_train: pd.DataFrame,
        x_val: pd.DataFrame,
        x_test: pd.DataFrame,
        numeric_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train and apply to val/test"""
    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_val_scaled = x_val.copy()
    x_test_scaled = x_test.copy()

    x_train_scaled[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_val_scaled[numeric_cols] = scaler.transform(x_val[numeric_cols])
    x_test_scaled[numeric_cols] = scaler.transform(x_test[numeric_cols])

    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train:pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        max_iter = max_iter,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model

def train_random_forest(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        random_state = random_state,
        class_weight = "balanced",
        n_jobs = -1,
    )
    model.fit(x_train, y_train)
    return model

def train_gradient_boosting(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        max_depth: int = 3,
        random_state: int = 42,
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        learning_rate = learning_rate,
        n_estimators = n_estimators,
        max_depth = max_depth,
        random_state = random_state
    )
    model.fit(x_train, y_train)
    return model

def evaluate_classifier(
        model,
        x: pd.DataFrame,
        y_true: pd.Series,
) -> Dict[str, float]:
    """Return common classification metrics."""
    y_pred = model.predict(x)

    if hasattr(model, "predict_proba"):
        # For binary classification: take probability of the positive class
        y_prob = model.predict_proba(x)[:, 1]
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        # For models without predict_proba but with decision_function
        y_score = model.decision_function(x)
        roc_auc = roc_auc_score(y_true, y_score)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }