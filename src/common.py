from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else WORKSPACE_ROOT / "HW-1"
LABEL_ORDER = ["business", "politics", "sports"]
_TOKENIZER = TreebankWordTokenizer()


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_nyt(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "nyt.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"nyt.csv is missing columns: {sorted(missing)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    return df


def load_ag(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "ag.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns:
        raise ValueError("ag.csv is missing the text column")
    df = df[["text"]].copy()
    df["text"] = df["text"].fillna("").astype(str)
    return df


def tokenize_text(text: str) -> list[str]:
    tokens = _TOKENIZER.tokenize(str(text).lower())
    return [token for token in tokens if any(ch.isalnum() for ch in token)]


def add_tokens(df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
    result = df.copy()
    result["tokens"] = result[column].map(tokenize_text)
    return result


def split_data(
    df: pd.DataFrame,
    label_column: str = "label",
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df[label_column],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[label_column],
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def class_distribution(df: pd.DataFrame, label_column: str = "label") -> dict[str, int]:
    counts = df[label_column].value_counts()
    return {str(label): int(counts[label]) for label in sorted(counts.index)}


def evaluate_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def save_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_text(path: Path | str, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def plot_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_array = np.asarray(matrix)
    plt.figure(figsize=(7.5, 6.0))
    sns.heatmap(
        matrix_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_metric_comparison(
    rows: list[dict[str, Any]],
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["method"] for row in rows]
    accuracy = [row["test_accuracy"] for row in rows]
    macro_f1 = [row["test_macro_f1"] for row in rows]
    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(8.5, 5.2))
    bars1 = plt.bar(x - width / 2, accuracy, width, label="Accuracy", color="#4C72B0")
    bars2 = plt.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#DD8452")
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Task 1: Bag-of-Words comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    packages = ["numpy", "pandas", "scipy", "sklearn", "nltk", "matplotlib", "seaborn"]
    for package in packages:
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostics only
            info[package] = f"unavailable: {exc}"
    return info


def save_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    result["predicted_label"] = classes[probabilities.argmax(axis=1)]
    result["confidence"] = probabilities.max(axis=1)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

