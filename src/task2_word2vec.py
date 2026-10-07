from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
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
    load_ag,
    load_nyt,
    plot_confusion_matrix,
    plot_metric_comparison,
    save_json,
    save_predictions,
    save_text,
    set_seed,
    split_data,
)


W2V_PARAMS: dict[str, Any] = {
    "vector_size": 100,
    "window": 5,
    "min_count": 5,
    "sg": 1,
    "negative": 5,
    "sample": 1e-3,
    "epochs": 5,
    "workers": 4,
    "seed": RANDOM_SEED,
}


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def train_or_load_word2vec(
    sentences: list[list[str]],
    model_path: Path,
    force_retrain: bool,
) -> Word2Vec:
    if model_path.exists() and not force_retrain:
        logging.info("Loading cached Word2Vec model: %s", model_path)
        return Word2Vec.load(str(model_path))
    logging.info("Training Word2Vec with %d documents", len(sentences))
    began = time.perf_counter()
    model = Word2Vec(sentences=sentences, **W2V_PARAMS)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logging.info("Saved Word2Vec model to %s in %.2f s", model_path, time.perf_counter() - began)
    return model


def mean_vectors(
    token_lists: list[list[str]],
    keyed_vectors: KeyedVectors,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, float]]:
    vectors: list[np.ndarray] = []
    total_tokens = 0
    covered_tokens = 0
    effective_counts: list[int] = []
    missing_types: set[str] = set()
    all_types: set[str] = set()

    for start in range(0, len(token_lists), batch_size):
        batch = token_lists[start : start + batch_size]
        for tokens in batch:
            total_tokens += len(tokens)
            known = [token for token in tokens if token in keyed_vectors]
            covered_tokens += len(known)
            effective_counts.append(len(known))
            all_types.update(tokens)
            missing_types.update(token for token in tokens if token not in keyed_vectors)
            if known:
                vectors.append(keyed_vectors.get_mean_vector(known, ignore_missing=True))
            else:
                vectors.append(np.zeros(int(keyed_vectors.vector_size), dtype=np.float32))

    matrix = np.vstack(vectors) if vectors else np.empty((0, int(keyed_vectors.vector_size)))
    stats = {
        "documents": int(len(token_lists)),
        "total_tokens": int(total_tokens),
        "covered_tokens": int(covered_tokens),
        "token_coverage": float(covered_tokens / total_tokens) if total_tokens else 0.0,
        "unique_token_types": int(len(all_types)),
        "oov_token_types": int(len(missing_types)),
        "oov_type_rate": float(len(missing_types) / len(all_types)) if all_types else 0.0,
        "mean_effective_tokens_per_document": float(np.mean(effective_counts)) if effective_counts else 0.0,
        "zero_vector_documents": int(sum(count == 0 for count in effective_counts)),
    }
    return matrix.astype(np.float32, copy=False), stats


def run_representation(
    method_name: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    representation: str,
    vector_stats: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], set[int]]:
    began = time.perf_counter()
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    classifier.fit(train_x, train_df["label"])
    val_pred = classifier.predict(val_x)
    test_pred = classifier.predict(test_x)
    test_prob = classifier.predict_proba(test_x)

    val_eval = evaluate_predictions(val_df["label"], val_pred, LABEL_ORDER)
    test_eval = evaluate_predictions(test_df["label"], test_pred, LABEL_ORDER)

    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    save_text(method_dir / "validation_classification_report.txt", val_eval["classification_report"])
    save_text(method_dir / "test_classification_report.txt", test_eval["classification_report"])
    plot_confusion_matrix(
        test_eval["confusion_matrix"],
        LABEL_ORDER,
        f"Task 2: {method_name} test confusion matrix",
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

    metrics: dict[str, Any] = {
        "method": method_name,
        "representation": representation,
        "vector_statistics": vector_stats,
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
    return metrics, set(int(value) for value in misclassified["row_id"].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Word2Vec and GloVe experiments")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "task2")
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=PROJECT_ROOT / "cache" / "glove" / "glove.6B.100d.txt",
    )
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "logs" / "task2.log")
    began = time.perf_counter()

    logging.info("Loading and tokenizing NYT")
    nyt = add_tokens(load_nyt(args.data_dir))
    train_df, val_df, test_df = split_data(nyt, seed=RANDOM_SEED)

    logging.info("Loading and tokenizing AG News")
    ag = add_tokens(load_ag(args.data_dir))

    dataset_summary = {
        "nyt_rows": int(len(nyt)),
        "nyt_class_distribution": class_distribution(nyt),
        "ag_rows": int(len(ag)),
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
    }
    save_json(args.output_dir / "dataset_summary.json", dataset_summary)

    representations: list[tuple[str, KeyedVectors, str]] = []

    if not args.glove_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {args.glove_path}. "
            "Download glove.6B.zip and extract glove.6B.100d.txt first."
        )
    logging.info("Loading GloVe vectors from %s", args.glove_path)
    glove_kv = KeyedVectors.load_word2vec_format(str(args.glove_path), binary=False, no_header=True)
    representations.append(("glove", glove_kv, "Pre-trained GloVe 6B 100d"))

    ag_model_path = args.output_dir / "models" / "word2vec_ag.model"
    ag_model = train_or_load_word2vec(ag["tokens"].tolist(), ag_model_path, args.force_retrain)
    representations.append(("word2vec_ag", ag_model.wv, "Word2Vec trained on AG News 100d"))

    nyt_sentences = train_df["tokens"].tolist()
    nyt_model_path = args.output_dir / "models" / "word2vec_nyt.model"
    nyt_model = train_or_load_word2vec(nyt_sentences, nyt_model_path, args.force_retrain)
    representations.append(("word2vec_nyt", nyt_model.wv, "Word2Vec trained on NYT training split 100d"))

    train_tokens = train_df["tokens"].tolist()
    val_tokens = val_df["tokens"].tolist()
    test_tokens = test_df["tokens"].tolist()

    rows: list[dict[str, Any]] = []
    error_sets: dict[str, set[int]] = {}
    for method_name, keyed_vectors, description in representations:
        logging.info("Building document vectors for %s", method_name)
        train_x, train_stats = mean_vectors(train_tokens, keyed_vectors)
        val_x, val_stats = mean_vectors(val_tokens, keyed_vectors)
        test_x, test_stats = mean_vectors(test_tokens, keyed_vectors)
        vector_stats = {
            "dimension": int(keyed_vectors.vector_size),
            "vocabulary_size": int(len(keyed_vectors)),
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        }
        logging.info(
            "%s coverage: train=%.4f validation=%.4f test=%.4f",
            method_name,
            train_stats["token_coverage"],
            val_stats["token_coverage"],
            test_stats["token_coverage"],
        )
        metrics, errors = run_representation(
            method_name,
            train_x,
            val_x,
            test_x,
            train_df,
            val_df,
            test_df,
            description,
            vector_stats,
            args.output_dir,
        )
        rows.append(metrics)
        error_sets[method_name] = errors
        logging.info(
            "%s: accuracy=%.4f macro_f1=%.4f errors=%d",
            method_name,
            metrics["test_accuracy"],
            metrics["test_macro_f1"],
            metrics["misclassified_count"],
        )

        model_cache = args.output_dir / "models" / f"{method_name}_document_vectors.pkl"
        with model_cache.open("wb") as handle:
            pickle.dump(
                {
                    "train": train_x,
                    "validation": val_x,
                    "test": test_x,
                    "stats": vector_stats,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    summary_table = pd.DataFrame(
        [
            {
                "method": row["method"],
                "representation": row["representation"],
                "validation_accuracy": row["validation_accuracy"],
                "validation_macro_f1": row["validation_macro_f1"],
                "test_accuracy": row["test_accuracy"],
                "test_macro_f1": row["test_macro_f1"],
                "vocabulary_size": row["vector_statistics"]["vocabulary_size"],
                "test_token_coverage": row["vector_statistics"]["test"]["token_coverage"],
                "test_oov_type_rate": row["vector_statistics"]["test"]["oov_type_rate"],
                "mean_effective_tokens": row["vector_statistics"]["test"]["mean_effective_tokens_per_document"],
                "misclassified_count": row["misclassified_count"],
                "elapsed_seconds": row["elapsed_seconds"],
            }
            for row in rows
        ]
    )
    summary_table.to_csv(args.output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    save_json(args.output_dir / "metrics_all.json", {"methods": rows, "w2v_parameters": W2V_PARAMS})
    save_json(args.output_dir / "environment.json", environment_info())
    plot_metric_comparison(rows, args.output_dir / "comparison.png")

    method_names = list(error_sets)
    error_analysis: dict[str, Any] = {}
    for name in method_names:
        error_analysis[f"{name}_errors"] = len(error_sets[name])
    error_analysis["common_to_all"] = len(set.intersection(*error_sets.values()))
    error_analysis["pairwise_common"] = {}
    for i, first in enumerate(method_names):
        for second in method_names[i + 1 :]:
            error_analysis["pairwise_common"][f"{first}__{second}"] = len(
                error_sets[first].intersection(error_sets[second])
            )
    save_json(args.output_dir / "error_analysis.json", error_analysis)

    logging.info("Task 2 completed in %.2f seconds", time.perf_counter() - began)


if __name__ == "__main__":
    main()
