import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm.auto import tqdm

from large_supervisor_models.train.config import TRAINED_MODEL_DIR, DEVICE, MAX_LEN, EVAL_BATCH_SIZE
from large_supervisor_models.training_data_loader import build_combined_dataset


# Dataset

class LSMDataset(Dataset):
    def __init__(self, data: List[Tuple[str, bool]], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(label), dtype=torch.long),
        }


def load_dataset(path: Path) -> List[Tuple[str, bool]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [(text, bool(label)) for text, label in raw]


# =========================
# Evaluation Core
# =========================


def run_inference(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    running_loss = 0.0
    num_batches = 0

    progress = tqdm(
        loader,
        desc="Evaluating",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            logits = outputs.logits
            loss = outputs.loss

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Accumulate
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            running_loss += loss.item()
            num_batches += 1

            # Live metrics
            running_acc = accuracy_score(all_labels, all_preds)
            avg_loss = running_loss / num_batches

            progress.set_postfix(
                {
                    "acc": f"{running_acc:.4f}",
                    "loss": f"{avg_loss:.4f}",
                    "samples": len(all_preds),
                }
            )

    return all_labels, all_preds, all_probs


# =========================
# Metrics
# =========================


def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    cm = confusion_matrix(labels, preds)

    return accuracy, precision, recall, f1, cm


def threshold_sweep(labels, probs):
    print("\n=== Threshold Sweep ===")

    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9]:
        preds = [1 if p > threshold else 0 for p in probs]

        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")

        print(
            f"Threshold={threshold:.1f} | "
            f"Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}"
        )


# =========================
# Entry Point
# =========================


def evaluate():
    print(f"Using device: {DEVICE}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_DIR).to(DEVICE)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    data = build_combined_dataset(test=True)

    dataset = LSMDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)

    # Run inference
    labels, preds, probs = run_inference(model, loader)

    # Metrics
    accuracy, precision, recall, f1, cm = compute_metrics(labels, preds)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Threshold sweep
    threshold_sweep(labels, probs)


# =========================

if __name__ == "__main__":
    evaluate()
