from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_XET_DISABLE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from common import (
    DEFAULT_DATA_DIR,
    LABEL_ORDER,
    PROJECT_ROOT,
    RANDOM_SEED,
    class_distribution,
    evaluate_predictions,
    load_nyt,
    plot_confusion_matrix,
    save_json,
    save_predictions,
    save_text,
    set_seed,
    split_data,
)


MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
LABEL2ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


class EncodedNewsDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][index], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.encodings["attention_mask"][index],
                dtype=torch.long,
            ),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = torch.tensor(
                self.encodings["token_type_ids"][index],
                dtype=torch.long,
            )
        return item


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


def set_reproducible_seed(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def encode_texts(tokenizer: Any, texts: list[str]) -> dict[str, list[list[int]]]:
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )


def make_loader(
    encodings: dict[str, list[list[int]]],
    labels: list[int],
    tokenizer: Any,
    shuffle: bool,
) -> DataLoader:
    dataset = EncodedNewsDataset(encodings, labels)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collator,
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="sum")
            total_loss += float(loss.item())
            total_samples += int(batch["labels"].size(0))
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())

    return (
        total_loss / max(total_samples, 1),
        np.concatenate(predictions),
        np.concatenate(probabilities),
        np.concatenate(labels),
    )


def full_token_length_statistics(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    lengths: list[int] = []
    batch_size = 256
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    array = np.asarray(lengths)
    return {
        "documents": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "over_max_length": int((array > MAX_LENGTH).sum()),
        "over_max_length_rate": float((array > MAX_LENGTH).mean()),
    }


def plot_training_history(history: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    f1_scores = [row["validation_macro_f1"] for row in history]

    figure, axis1 = plt.subplots(figsize=(8.5, 5.2))
    axis1.plot(epochs, losses, marker="o", color="#4C72B0", label="Training loss")
    axis1.set_xlabel("Epoch")
    axis1.set_ylabel("Training loss", color="#4C72B0")
    axis1.tick_params(axis="y", labelcolor="#4C72B0")
    axis1.grid(alpha=0.25)

    axis2 = axis1.twinx()
    axis2.plot(epochs, f1_scores, marker="s", color="#DD8452", label="Validation Macro-F1")
    axis2.set_ylabel("Validation Macro-F1", color="#DD8452")
    axis2.tick_params(axis="y", labelcolor="#DD8452")
    axis2.set_ylim(min(0.0, min(f1_scores) - 0.02), 1.0)

    lines1, labels1 = axis1.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("BERT fine-tuning history")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def environment_info(device: torch.device) -> dict[str, Any]:
    import sklearn
    import transformers

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: BERT fine-tuning")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "task3")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    if args.epochs != NUM_EPOCHS:
        raise ValueError(f"This experiment requires exactly {NUM_EPOCHS} epochs")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"The recorded experiment configuration uses batch_size={BATCH_SIZE}")

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "logs" / "task3.log")
    set_reproducible_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    logging.info("Loading tokenizer and model: %s", args.model_name)

    cache_dir = PROJECT_ROOT / "cache" / "huggingface"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_ORDER),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        cache_dir=cache_dir,
    ).to(device)

    df = load_nyt(args.data_dir)
    train_df, val_df, test_df = split_data(df, seed=RANDOM_SEED)
    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()
    train_labels = [LABEL2ID[label] for label in train_df["label"]]
    val_labels = [LABEL2ID[label] for label in val_df["label"]]
    test_labels = [LABEL2ID[label] for label in test_df["label"]]

    logging.info("Tokenizing datasets with max_length=%d", MAX_LENGTH)
    train_encodings = encode_texts(tokenizer, train_texts)
    val_encodings = encode_texts(tokenizer, val_texts)
    test_encodings = encode_texts(tokenizer, test_texts)

    train_loader = make_loader(train_encodings, train_labels, tokenizer, shuffle=True)
    val_loader = make_loader(val_encodings, val_labels, tokenizer, shuffle=False)
    test_loader = make_loader(test_encodings, test_labels, tokenizer, shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_macro_f1 = -1.0
    training_started = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            batch_size = int(batch["labels"].size(0))
            epoch_loss += float(loss.item()) * batch_size
            epoch_samples += batch_size
            if step % 100 == 0:
                logging.info(
                    "Epoch %d step %d/%d loss=%.4f",
                    epoch,
                    step,
                    len(train_loader),
                    epoch_loss / epoch_samples,
                )

        train_loss = epoch_loss / max(epoch_samples, 1)
        val_loss, val_pred_ids, _, val_true_ids = evaluate_model(model, val_loader, device)
        val_true_labels = [ID2LABEL[int(value)] for value in val_true_ids]
        val_pred_labels = [ID2LABEL[int(value)] for value in val_pred_ids]
        val_evaluation = evaluate_predictions(val_true_labels, val_pred_labels, LABEL_ORDER)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_evaluation["accuracy"]),
            "validation_macro_f1": float(val_evaluation["macro_f1"]),
            "elapsed_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(row)
        logging.info(
            "Epoch %d complete: train_loss=%.4f val_loss=%.4f val_accuracy=%.4f val_macro_f1=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_evaluation["accuracy"],
            val_evaluation["macro_f1"],
        )

        if val_evaluation["macro_f1"] > best_validation_macro_f1:
            best_validation_macro_f1 = float(val_evaluation["macro_f1"])
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )

    if best_state is None:
        raise RuntimeError("No best model state was captured")
    model.load_state_dict(best_state)
    model.to(device)

    test_loss, test_pred_ids, test_probabilities, test_true_ids = evaluate_model(
        model,
        test_loader,
        device,
    )
    test_true_labels = [ID2LABEL[int(value)] for value in test_true_ids]
    test_pred_labels = [ID2LABEL[int(value)] for value in test_pred_ids]
    test_evaluation = evaluate_predictions(test_true_labels, test_pred_labels, LABEL_ORDER)

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_training_history(history, args.output_dir / "training_curves.png")
    plot_confusion_matrix(
        test_evaluation["confusion_matrix"],
        LABEL_ORDER,
        "Task 3: BERT-base-uncased test confusion matrix",
        args.output_dir / "test_confusion_matrix.png",
    )
    save_text(
        args.output_dir / "validation_classification_report.txt",
        evaluate_predictions(
            [ID2LABEL[int(value)] for value in val_true_ids],
            [ID2LABEL[int(value)] for value in val_pred_ids],
            LABEL_ORDER,
        )["classification_report"],
    )
    save_text(
        args.output_dir / "test_classification_report.txt",
        test_evaluation["classification_report"],
    )

    prediction_frame = test_df[["text", "label"]].reset_index(drop=True)
    save_predictions(
        prediction_frame,
        test_probabilities,
        np.asarray(LABEL_ORDER),
        args.output_dir / "predictions.csv",
    )
    predicted = pd.DataFrame(
        {
            "row_id": test_df.index.to_numpy(),
            "text": test_df["text"].to_numpy(),
            "true_label": test_true_labels,
            "predicted_label": test_pred_labels,
            "confidence": test_probabilities.max(axis=1),
        }
    )
    misclassified = predicted[predicted["true_label"] != predicted["predicted_label"]].copy()
    misclassified.sort_values("confidence", ascending=False, inplace=True)
    misclassified.to_csv(args.output_dir / "misclassified_examples.csv", index=False, encoding="utf-8-sig")

    truncation_stats = {
        "max_length": MAX_LENGTH,
        "train": full_token_length_statistics(tokenizer, train_texts),
        "validation": full_token_length_statistics(tokenizer, val_texts),
        "test": full_token_length_statistics(tokenizer, test_texts),
    }
    save_json(args.output_dir / "truncation_statistics.json", truncation_stats)

    metrics = {
        "model_name": args.model_name,
        "max_length": MAX_LENGTH,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_evaluation["accuracy"]),
        "test_macro_f1": float(test_evaluation["macro_f1"]),
        "test_confusion_matrix": test_evaluation["confusion_matrix"],
        "test_labels": LABEL_ORDER,
        "misclassified_count": int(len(misclassified)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": float(time.perf_counter() - training_started),
    }
    save_json(args.output_dir / "metrics.json", metrics)
    save_json(args.output_dir / "environment.json", environment_info(device))
    save_json(
        args.output_dir / "dataset_summary.json",
        {
            "rows": int(len(df)),
            "class_distribution": class_distribution(df),
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
        },
    )

    logging.info(
        "Task 3 complete: best_epoch=%d accuracy=%.4f macro_f1=%.4f errors=%d",
        best_epoch,
        test_evaluation["accuracy"],
        test_evaluation["macro_f1"],
        len(misclassified),
    )


if __name__ == "__main__":
    main()


