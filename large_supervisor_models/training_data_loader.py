import os
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)


# Utility functions

def _save_json(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_sample(text: str, is_harmful: bool) -> Tuple[str, bool]:
    return (text.strip(), bool(is_harmful))


# Dataset loaders

def load_lmsys_toxic_chat(test: bool = False) -> List[Dict]:
    split = "test" if test else "train"
    dataset_name = "lmsys/toxic-chat"
    dataset = load_dataset(dataset_name, "toxicchat1123", split=split)

    processed = []
    for i, row in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
        text = row["model_output"]
        toxicity = row["toxicity"] or row["jailbreaking"]
        if not text:
            logging.warning(f"------ Empty text for row number {i} in {dataset_name}")
            continue

        processed.append(_normalize_sample(text, toxicity))

    return processed


def load_toxigen(test: bool = False) -> List[Dict]:
    split = "test" if test else "train"
    dataset_name = "toxigen/toxigen-data"
    dataset = load_dataset(dataset_name, "annotated", split=split)

    processed = []
    for i, row in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
        text = row["text"]
        toxicity_human = row["toxicity_human"]
        toxicity = toxicity_human >= 3
        if not text:
            logging.warning(f"------ Empty text for row number {i} in {dataset_name}")
            continue

        processed.append(_normalize_sample(text, toxicity))

    return processed


def load_jigsaw(test: bool = False) -> List[Dict]:
    split = "test" if test else "train"
    dataset_name = "TurkuNLP/jigsaw_toxicity_pred_fi"
    dataset = load_dataset(dataset_name, split=split)

    processed = []
    for i, row in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
        text = row["text"]
        toxicity = (
            row["label_identity_attack"]
            or row["label_insult"]
            or row["label_obscene"]
            or row["label_severe_toxicity"]
            or row["label_threat"]
            or row["label_toxicity"]
        )
        if not text:
            logging.warning(f"------ Empty text for row number {i} in {dataset_name}")
            continue

        processed.append(_normalize_sample(text, toxicity))

    return processed


def load_real_toxicity_prompts(test: bool = False) -> List[Dict]:
    # Same subset for train and test
    dataset_name = "allenai/real-toxicity-prompts"
    dataset = load_dataset(dataset_name, split="train")

    processed = []
    for i, row in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
        row_data = row["continuation"]
        text = row_data["text"]
        toxicity_values = (
            row_data["profanity"],
            row_data["sexually_explicit"],
            row_data["identity_attack"],
            row_data["flirtation"],
            row_data["threat"],
            row_data["insult"],
            row_data["severe_toxicity"],
            row_data["toxicity"],
        )
        toxicity = any(float(value or 0) > 0.5 for value in toxicity_values)
        if not text:
            logging.warning(f"------ Empty text for row number {i} in {dataset_name}")
            continue

        processed.append(_normalize_sample(text, toxicity))

    return processed


def load_aegis(test: bool = False) -> List[Dict]:
    split = "test" if test else "train"
    dataset_name = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
    dataset = load_dataset(dataset_name, split=split)

    processed = []
    for i, row in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
        text = row["response"]
        is_harmful = row["response_label"] == "unsafe"
        if not text:
            logging.warning(f"------ Empty text for row number {i} in {dataset_name}")
            continue

        processed.append(_normalize_sample(text, is_harmful))

    return processed


def load_harmful_dataset(test: bool = False) -> List[Dict]:
    # Same subset for train and test
    dataset_name = "LLM-LAT/harmful-dataset"
    dataset = load_dataset(dataset_name, split="train")

    processed = []
    for i, row in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
        text = row["rejected"]
        if not text:
            logging.warning(f"------ Empty text for row number {i} in {dataset_name}")
            continue

        processed.append(_normalize_sample(text, True))

    return processed


# Combine datasets

def build_combined_dataset(test: bool = False):
    cache_name = "combined-dataset-test.json" if test else "combined-dataset.json"
    cache_path = CACHE_DIR / cache_name
    if os.path.exists(cache_path):
        logging.info("Loading combined dataset from cache: %s", cache_path)
        return _load_json(str(cache_path))

    datasets = [
        load_lmsys_toxic_chat(test=test),
        load_toxigen(test=test),
        load_jigsaw(test=test),
        load_real_toxicity_prompts(test=test),
        load_aegis(test=test),
        load_harmful_dataset(test=test),
    ]
    combined = [sample for dataset in datasets
                for sample in dataset]
    logging.info("Total combined samples: %d", len(combined))

    _save_json(str(cache_path), combined)
    return combined


if __name__ == "__main__":
    data = build_combined_dataset()
    logging.info("Dataset ready with %d samples.", len(data))
