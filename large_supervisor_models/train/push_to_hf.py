import os

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise EnvironmentError(
        "HF_TOKEN environment variable not set. Export your Hugging Face token as HF_TOKEN."
    )


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import create_repo

from large_supervisor_models.train.config import TRAINED_MODEL_DIR, DEVICE


# Config
HF_REPO_ID = "prane-eth/Large-Supervisor-Model"
PRIVATE = False


# Load model

def load_local_model():
    print("Loading local model...")

    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODEL_DIR)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# Sanity check

def sanity_check(tokenizer, model):
    print("Running sanity check...")

    text = "This is a test sentence."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    print("Output probabilities:", probs.cpu().numpy())


# Push to HF

def push_to_hub(tokenizer, model):
    # Read token from environment. Prefer HF_TOKEN, fallback to HUGGINGFACE_HUB_TOKEN.
    print(f"Creating repo: {HF_REPO_ID}")

    create_repo(
        repo_id=HF_REPO_ID,
        private=PRIVATE,
        exist_ok=True,
        token=token,
    )

    print("Pushing tokenizer...")
    tokenizer.push_to_hub(HF_REPO_ID, use_auth_token=token)

    print("Pushing model...")
    model.push_to_hub(HF_REPO_ID, use_auth_token=token)

    print(f"Done. Model available at: https://huggingface.co/{HF_REPO_ID}")


# Main

def main():
    tokenizer, model = load_local_model()
    sanity_check(tokenizer, model)
    push_to_hub(tokenizer, model)


if __name__ == "__main__":
    main()
