from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from common import (
    DEFAULT_DATA_DIR,
    LABEL_ORDER,
    PROJECT_ROOT,
    RANDOM_SEED,
    add_tokens,
    class_distribution,
    environment_info,
    evaluate_predictions,
    load_nyt,
    plot_confusion_matrix,
    plot_metric_comparison,
    save_json,
    save_predictions,
    save_text,
    set_seed,
    split_data,
)


def identity(tokens: list[str]) -> list[str]:
    return tokens


def token_statistics(df: pd.DataFrame) -> dict[str, float]:
    lengths = df["tokens"].map(len)
    return {
        "count": int(lengths.count()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "max": int(lengths.max()),
        "p95": float(lengths.quantile(0.95)),
        "empty_documents": int((lengths == 0).sum()),
    }


def top_features(
    vectorizer: CountVectorizer,
    classifier: LogisticRegression,
    labels: list[str],
    top_k: int = 20,
) -> pd.DataFrame:
    terms = np.asarray(vectorizer.get_feature_names_out())
    rows: list[dict[str, object]] = []
    for class_index, label in enumerate(labels):
        coefficients = classifier.coef_[class_index]
        order = np.argsort(coefficients)[::-1][:top_k]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "class": label,
                    "rank": rank,
                    "term": terms[feature_index],
                    "coefficient": float(coefficients[feature_index]),
                }
            )
    return pd.DataFrame(rows)


def run_method(
    method_name: str,
    binary: bool,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, object], set[int]]:
    began = time.perf_counter()
    vectorizer = CountVectorizer(
        analyzer=identity,
        lowercase=False,
        token_pattern=None,
        binary=binary,
        dtype=np.float64,
    )
    x_train = vectorizer.fit_transform(train_df["tokens"])
    x_val = vectorizer.transform(val_df["tokens"])
    x_test = vectorizer.transform(test_df["tokens"])

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    classifier.fit(x_train, train_df["label"])

    val_pred = classifier.predict(x_val)
    test_pred = classifier.predict(x_test)
    val_prob = classifier.predict_proba(x_val)
    test_prob = classifier.predict_proba(x_test)

    val_eval = evaluate_predictions(val_df["label"], val_pred, LABEL_ORDER)
    test_eval = evaluate_predictions(test_df["label"], test_pred, LABEL_ORDER)

    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    save_text(
        method_dir / "validation_classification_report.txt",
        val_eval["classification_report"],
    )
    save_text(
        method_dir / "test_classification_report.txt",
        test_eval["classification_report"],
    )
    plot_confusion_matrix(
        test_eval["confusion_matrix"],
        LABEL_ORDER,
        f"Task 1: {method_name} test confusion matrix",
        method_dir / "test_confusion_matrix.png",
    )

    prediction_frame = test_df[["text", "label"]].reset_index(drop=True)
    save_predictions(prediction_frame, test_prob, classifier.classes_, method_dir / "predictions.csv")

    predicted = pd.DataFrame(
        {
            "row_id": test_df.index.to_numpy(),
            "text": test_df["text"].to_numpy(),
            "true_label": test_df["label"].to_numpy(),
            "predicted_label": test_pred,
            "confidence": test_prob.max(axis=1),
        }
    )
    misclassified = predicted[predicted["true_label"] != predicted["predicted_label"]].copy()
    misclassified.sort_values("confidence", ascending=False, inplace=True)
    misclassified.to_csv(method_dir / "misclassified_examples.csv", index=False, encoding="utf-8-sig")

    top_features(vectorizer, classifier, LABEL_ORDER).to_csv(
        method_dir / "top_features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "method": method_name,
        "representation": "binary bag-of-words" if binary else "word frequency",
        "vectorizer": {
            "binary": binary,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "matrix_shape_train": list(x_train.shape),
            "matrix_shape_test": list(x_test.shape),
            "train_nonzero": int(x_train.nnz),
            "test_nonzero": int(x_test.nnz),
        },
        "classifier": {
            "type": "LogisticRegression",
            "C": 1.0,
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": RANDOM_SEED,
        },
        "validation_accuracy": val_eval["accuracy"],
        "validation_macro_f1": val_eval["macro_f1"],
        "test_accuracy": test_eval["accuracy"],
        "test_macro_f1": test_eval["macro_f1"],
        "test_confusion_matrix": test_eval["confusion_matrix"],
        "test_labels": LABEL_ORDER,
        "misclassified_count": int(len(misclassified)),
        "elapsed_seconds": float(time.perf_counter() - began),
    }
    save_json(method_dir / "metrics.json", metrics)

    misclassified_ids = set(int(value) for value in misclassified["row_id"].tolist())
    return metrics, misclassified_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Bag-of-Words experiments")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "task1")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    df = load_nyt(args.data_dir)
    df = add_tokens(df)
    train_df, val_df, test_df = split_data(df, seed=RANDOM_SEED)

    summary = {
        "raw_rows": int(len(df)),
        "class_distribution": class_distribution(df),
        "token_statistics": token_statistics(df),
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_class_distribution": {
            "train": class_distribution(train_df),
            "validation": class_distribution(val_df),
            "test": class_distribution(test_df),
        },
        "split_seed": RANDOM_SEED,
    }
    save_json(args.output_dir / "dataset_summary.json", summary)

    split_manifest = df[["label"]].copy()
    split_manifest["token_count"] = df["tokens"].map(len)
    split_manifest["split"] = "unassigned"
    split_manifest.loc[train_df.index, "split"] = "train"
    split_manifest.loc[val_df.index, "split"] = "validation"
    split_manifest.loc[test_df.index, "split"] = "test"
    split_manifest.reset_index(names="row_id").to_csv(
        args.output_dir / "split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    methods = [
        ("binary_bow", True),
        ("word_frequency", False),
    ]
    rows: list[dict[str, object]] = []
    misclassified_sets: dict[str, set[int]] = {}
    for method_name, binary in methods:
        print(f"Running {method_name} ...", flush=True)
        metrics, misclassified_ids = run_method(
            method_name,
            binary,
            train_df,
            val_df,
            test_df,
            args.output_dir,
        )
        rows.append(metrics)
        misclassified_sets[method_name] = misclassified_ids
        print(
            f"{method_name}: accuracy={metrics['test_accuracy']:.4f}, "
            f"macro_f1={metrics['test_macro_f1']:.4f}, "
            f"vocab={metrics['vectorizer']['vocabulary_size']}",
            flush=True,
        )

    summary_table = pd.DataFrame(
        [
            {
                "method": row["method"],
                "validation_accuracy": row["validation_accuracy"],
                "validation_macro_f1": row["validation_macro_f1"],
                "test_accuracy": row["test_accuracy"],
                "test_macro_f1": row["test_macro_f1"],
                "vocabulary_size": row["vectorizer"]["vocabulary_size"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for row in rows
        ]
    )
    summary_table.to_csv(args.output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    save_json(args.output_dir / "metrics_all.json", {"methods": rows})
    plot_metric_comparison(rows, args.output_dir / "comparison.png")

    overlap = misclassified_sets["binary_bow"].intersection(misclassified_sets["word_frequency"])
    analysis = {
        "binary_only_errors": len(misclassified_sets["binary_bow"] - misclassified_sets["word_frequency"]),
        "word_frequency_only_errors": len(misclassified_sets["word_frequency"] - misclassified_sets["binary_bow"]),
        "common_errors": len(overlap),
        "binary_error_count": len(misclassified_sets["binary_bow"]),
        "word_frequency_error_count": len(misclassified_sets["word_frequency"]),
    }
    save_json(args.output_dir / "error_overlap.json", analysis)
    save_json(args.output_dir / "environment.json", environment_info())

    runtime = time.perf_counter() - started
    print(f"Task 1 completed in {runtime:.2f} seconds.", flush=True)


if __name__ == "__main__":
    main()


