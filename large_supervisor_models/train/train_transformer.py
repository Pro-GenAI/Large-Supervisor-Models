import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from large_supervisor_models.training_data_loader import build_combined_dataset
from large_supervisor_models.train.config import (
    TRAINED_MODEL_DIR,
    CHECKPOINT_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    GRAD_ACCUM,
    EPOCHS,
    MAX_LEN,
    USE_FP16,
    DEVICE,
)

# ----------------------------------
# Dataset
# ----------------------------------


def prepare_dataset():
    data = build_combined_dataset()
    random.shuffle(data)

    texts = [x[0] if x[0] is not None else "" for x in data]
    labels = [int(x[1]) for x in data]

    ds = Dataset.from_dict({"text": texts, "label": labels})

    split = ds.train_test_split(test_size=0.1)
    return split["train"], split["test"]


def tokenize_fn(examples):
    texts = [t if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, truncation=True, max_length=MAX_LEN)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def get_last_checkpoint():
    if not CHECKPOINT_DIR.exists():
        return None

    checkpoints = list(CHECKPOINT_DIR.glob("checkpoint-*"))
    if not checkpoints:
        return None

    return str(sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1])


def train():
    global tokenizer

    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds, val_ds = prepare_dataset()

    # Tokenization with column cleanup (CRITICAL FIX)
    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=2,
        remove_columns=["text"],
    )

    val_ds = val_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=2,
        remove_columns=["text"],
    )

    # Debug check (ensures correctness)
    print("Sample tokenized entry:", train_ds[0])

    # Set PyTorch format
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # IMPORTANT: Do NOT manually move model or compile
    # Trainer handles device placement

    training_args = TrainingArguments(
        output_dir=str(TRAINED_MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        # PERFORMANCE
        fp16=USE_FP16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # LOGGING
        logging_steps=50,
        logging_dir=str(CHECKPOINT_DIR / "logs"),
        # EVAL + SAVE
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # OPTIM
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Verify device placement
    print("Model device:", next(model.parameters()).device)

    # Resume logic
    last_ckpt = get_last_checkpoint()
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("Starting fresh training")
        trainer.train()

    trainer.save_model(str(TRAINED_MODEL_DIR))
    tokenizer.save_pretrained(str(TRAINED_MODEL_DIR))


if __name__ == "__main__":
    train()
