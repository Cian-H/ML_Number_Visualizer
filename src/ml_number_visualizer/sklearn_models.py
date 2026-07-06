import joblib
import numpy as np
from pathlib import Path

from loguru import logger
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_number_visualizer.utils import extract_numpy_data

_N_STEPS = 10
_SNAPSHOT_BASE = Path("./checkpoints/snapshots")
_MODELS_DIR = Path("./models")


# ---------------------------------------------------------------------------
# Snapshot strategies — each returns (paths, frame_labels)
# ---------------------------------------------------------------------------

def _dump(model, snapshot_dir: Path, step: int) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"step_{step:03d}.joblib"
    joblib.dump(model, path)
    return path


def _train_warm_start_trees(
    model,
    X: np.ndarray,
    y: np.ndarray,
    snapshot_dir: Path,
    trees_per_step: int = 5,
    label_noun: str = "Trees",
) -> tuple[list[Path], list[str]]:
    """Incrementally grow an ensemble via warm_start (RF, ExtraTrees, GradientBoosting)."""
    paths, labels = [], []
    for step in range(_N_STEPS):
        model.n_estimators = trees_per_step * (step + 1)
        model.fit(X, y)
        paths.append(_dump(model, snapshot_dir, step))
        labels.append(f"{model.n_estimators} {label_noun}")
        logger.debug(f"  → {labels[-1]}")
    return paths, labels


def _train_regularization_path(
    model_class,
    fixed_kwargs: dict,
    C_values: list[float],
    X: np.ndarray,
    y: np.ndarray,
    snapshot_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Train a fresh model at each point on a regularization path (varying C).

    Low C = strong regularisation (underfitting).
    High C = weak regularisation (overfitting / complex boundary).
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = [], []
    for step, C in enumerate(C_values):
        model = model_class(C=C, **fixed_kwargs)
        model.fit(X, y)
        paths.append(_dump(model, snapshot_dir, step))
        labels.append(f"C={C:.3g}")
        logger.debug(f"  → C={C:.3g}")
    return paths, labels


def _train_partial_fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    snapshot_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Stream data in equal batches via partial_fit (GaussianNB, SGDClassifier, etc.)."""
    classes = np.unique(y)
    batches = np.array_split(np.arange(len(X)), _N_STEPS)
    paths, labels = [], []
    for step, idx in enumerate(batches):
        model.partial_fit(X[idx], y[idx], classes=classes)
        paths.append(_dump(model, snapshot_dir, step))
        labels.append(f"Batch {step + 1}/{_N_STEPS}")
        logger.debug(f"  → Batch {step + 1}/{_N_STEPS} ({len(idx)} samples)")
    return paths, labels


def _train_decision_tree(
    X: np.ndarray, y: np.ndarray, snapshot_dir: Path
) -> tuple[list[Path], list[str]]:
    """Grow decision tree from depth 1 to _N_STEPS (shows overfitting progression)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = [], []
    for step in range(_N_STEPS):
        depth = step + 1
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X, y)
        paths.append(_dump(model, snapshot_dir, step))
        labels.append(f"Depth {depth}")
        logger.debug(f"  → Depth {depth}")
    return paths, labels


def _train_knn(
    X: np.ndarray, y: np.ndarray, snapshot_dir: Path
) -> tuple[list[Path], list[str]]:
    """Vary k from large (smooth/underfitting) to small (complex/overfitting).

    This traverses the bias-variance frontier of the KNN decision boundary.
    """
    k_sequence = [100, 75, 50, 35, 25, 15, 9, 5, 3, 1][:_N_STEPS]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths, labels = [], []
    for step, k in enumerate(k_sequence):
        k = min(k, len(X) - 1)  # clamp for small datasets
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        paths.append(_dump(model, snapshot_dir, step))
        labels.append(f"K={k}")
        logger.debug(f"  → K={k}")
    return paths, labels


# ---------------------------------------------------------------------------
# Registry of all sklearn models
# ---------------------------------------------------------------------------

def train_sklearn(
    train_loader, val_loader, test_loader
) -> dict[str, tuple[list[Path], list[str]]]:
    """Train all sklearn classifiers with incremental snapshots for video generation.

    Returns:
        dict mapping model_key → (snapshot_paths, frame_labels)
    """
    logger.info("Fetching dataset for Scikit-Learn...")
    X_train, y_train = extract_numpy_data(train_loader)
    extract_numpy_data(val_loader)   # consumed but not used
    extract_numpy_data(test_loader)  # consumed but not used

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, tuple[list[Path], list[str]]] = {}

    # -- Random Forest (bagging, axis-aligned trees) -----------------------
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=5, max_depth=15, warm_start=True, n_jobs=-1)
    p, l = _train_warm_start_trees(rf, X_train, y_train, _SNAPSHOT_BASE / "sklearn_rf")
    joblib.dump(rf, _MODELS_DIR / "sklearn_rf.joblib")
    snapshots["sklearn_rf"] = (p, l)

    # -- Extra Trees (extremely randomised forests) ------------------------
    logger.info("Training Extra Trees...")
    et = ExtraTreesClassifier(n_estimators=5, max_depth=15, warm_start=True, n_jobs=-1)
    p, l = _train_warm_start_trees(et, X_train, y_train, _SNAPSHOT_BASE / "sklearn_extra_trees")
    joblib.dump(et, _MODELS_DIR / "sklearn_extra_trees.joblib")
    snapshots["sklearn_extra_trees"] = (p, l)

    # -- Gradient Boosting (sequential boosting) ---------------------------
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=5, warm_start=True)
    p, l = _train_warm_start_trees(
        gb, X_train, y_train, _SNAPSHOT_BASE / "sklearn_grad_boost", label_noun="Estimators"
    )
    joblib.dump(gb, _MODELS_DIR / "sklearn_grad_boost.joblib")
    snapshots["sklearn_grad_boost"] = (p, l)

    # -- SVM with RBF kernel (regularisation path) -------------------------
    # C controls the margin width: small C = wide margin (underfitting),
    # large C = narrow margin (overfitting). More interesting than data volume.
    logger.info("Training SVM (RBF kernel, regularisation path)...")
    svm_C = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    p, l = _train_regularization_path(
        SVC,
        {"kernel": "rbf", "probability": True, "gamma": "scale", "random_state": 42},
        svm_C,
        X_train, y_train,
        _SNAPSHOT_BASE / "sklearn_svm",
    )
    joblib.dump(joblib.load(p[-1]), _MODELS_DIR / "sklearn_svm.joblib")
    snapshots["sklearn_svm"] = (p, l)

    # -- Logistic Regression (regularisation path) -------------------------
    logger.info("Training Logistic Regression (regularisation path)...")
    lr_C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    p, l = _train_regularization_path(
        LogisticRegression,
        {"solver": "saga", "max_iter": 2000, "random_state": 42},
        lr_C,
        X_train, y_train,
        _SNAPSHOT_BASE / "sklearn_logistic",
    )
    joblib.dump(joblib.load(p[-1]), _MODELS_DIR / "sklearn_logistic.joblib")
    snapshots["sklearn_logistic"] = (p, l)

    # -- Gaussian Naive Bayes (online Bayesian update) ---------------------
    logger.info("Training Gaussian Naive Bayes...")
    gnb = GaussianNB()
    p, l = _train_partial_fit(gnb, X_train, y_train, _SNAPSHOT_BASE / "sklearn_naive_bayes")
    joblib.dump(gnb, _MODELS_DIR / "sklearn_naive_bayes.joblib")
    snapshots["sklearn_naive_bayes"] = (p, l)

    # -- Decision Tree (depth progression shows overfitting) ---------------
    logger.info("Training Decision Tree (depth progression)...")
    p, l = _train_decision_tree(X_train, y_train, _SNAPSHOT_BASE / "sklearn_decision_tree")
    joblib.dump(joblib.load(p[-1]), _MODELS_DIR / "sklearn_decision_tree.joblib")
    snapshots["sklearn_decision_tree"] = (p, l)

    # -- K-Nearest Neighbours (k: high→low = smooth→complex) ---------------
    logger.info("Training K-Nearest Neighbours (varying K)...")
    p, l = _train_knn(X_train, y_train, _SNAPSHOT_BASE / "sklearn_knn")
    # Production model uses k=5 (good default for most datasets)
    final_knn = KNeighborsClassifier(n_neighbors=min(5, len(X_train) - 1))
    final_knn.fit(X_train, y_train)
    joblib.dump(final_knn, _MODELS_DIR / "sklearn_knn.joblib")
    snapshots["sklearn_knn"] = (p, l)

    return snapshots
