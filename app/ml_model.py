"""Machine-learning model for student employability prediction.

Trains a Random Forest classifier on the Student-Employability dataset using
GridSearchCV for hyperparameter tuning, 5-fold cross-validation, and an 80/20
train-test split.  The trained model and preprocessing pipeline are saved to
disk with joblib so the application loads them without retraining on every
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .employability_form import EMPLOYABILITY_FIELDS, MLPredictionResult

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Student-Employability-Datasets.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "employability_rf.joblib"
METRICS_PATH = MODEL_DIR / "employability_metrics.joblib"

FEATURE_COLUMNS = [f.key for f in EMPLOYABILITY_FIELDS]
TARGET_COLUMN = "CLASS"


@dataclass
class ModelMetrics:
    """Holds measured evaluation metrics from the trained model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_best_score: float
    confusion_matrix: list[list[int]]
    classification_report: str
    train_size: int
    test_size: int
    best_params: dict = field(default_factory=dict)


def train_model() -> ModelMetrics:
    """Train, evaluate and persist the Random Forest model."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])

    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, average="weighted")),
        recall=float(recall_score(y_test, y_pred, average="weighted")),
        f1=float(f1_score(y_test, y_pred, average="weighted")),
        cv_best_score=float(grid.best_score_),
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        classification_report=classification_report(y_test, y_pred),
        train_size=len(X_train),
        test_size=len(X_test),
        best_params=dict(grid.best_params_),
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return metrics


_cached_model: Pipeline | None = None
_cached_metrics: ModelMetrics | None = None


def load_model() -> Pipeline:
    """Load the persisted model, training on first call if needed."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not MODEL_PATH.exists():
        train_model()
    _cached_model = joblib.load(MODEL_PATH)
    return _cached_model


def load_metrics() -> ModelMetrics:
    """Load persisted metrics, training on first call if needed."""
    global _cached_metrics
    if _cached_metrics is not None:
        return _cached_metrics
    if not METRICS_PATH.exists():
        train_model()
    _cached_metrics = joblib.load(METRICS_PATH)
    return _cached_metrics


def predict(features: dict[str, int]) -> MLPredictionResult:
    """Predict employability from a dict of column-name -> rating (1-5).

    Returns the predicted class label and model-estimated probabilities.
    """
    model = load_model()
    row = pd.DataFrame(
        [[float(features.get(col, 3)) for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    predicted_class = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = model.classes_

    probabilities: dict[str, float] = {}
    for cls, p in zip(classes, proba):
        probabilities[str(cls)] = round(float(p), 4)

    confidence = float(probabilities.get(str(predicted_class), 0.0))

    return MLPredictionResult(
        predicted_class=str(predicted_class),
        probabilities=probabilities,
        confidence=confidence,
    )