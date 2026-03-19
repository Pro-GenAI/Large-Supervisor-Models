# 🛡️ LaSuMo: Large Supervisor Models (LSMs)

> **Real-time AI safety — not bolted on after, but woven in as it happens.**

<p align="center">
<!-- <img src="./assets/project_banner.jpg" alt="Project banner" height="360px"/> -->
<!-- $ convert project_banner.png -resize 600x319 project_banner.jpg -->
<!-- <img src="./assets/project_logo.jpg" alt="Project logo" width="270px"/> -->
<!-- $ convert logo_large.png -resize 270x270 project_logo.jpg -->
<img src="https://raw.githubusercontent.com/Pro-GenAI/Large-Supervisor-Models/main/assets/Banner.jpg" alt="Banner image" height="400"/>
</p>

## The Problem with AI Safety Today

Most content moderation happens *after* an LLM finishes speaking. By then, the damage is done — harmful content has already been generated, streamed, and delivered to the user. Guardrails that wrap around LLMs are slow instruments that can analyze the input and final output, but not segments in the reasoning or output streams to interrupt harmful content as soon as it starts to emerge.

**LSM changes that.**

## What is an LSM?

A **Large Supervisor Model** is a lightweight, purpose-built model that runs *in parallel* with any LLM — reading the token output stream in real time and intervening the moment it detects something harmful. It doesn't wait. It doesn't post-process. It watches every token as it arrives and acts instantly.

Think of it as a co-pilot that never blinks.

```
LLM ──── token stream ────▶ LSM ──── abstain / INTERRUPT ──▶ Client
                                              ▲
                                    Running in parallel,
                                    always watching
```

---

## Key Features

| Feature | Description |
|---|---|
| ⚡ **Real-time interruption** | Fires an interrupt signal mid-stream the moment harmful content is detected |
| 🔇 **Silent feedback** | Logs, reports, or flags content in the background without disrupting normal responses |
| 🪶 **Lightweight by design** | Small and fast — built to shadow any LLM without meaningful overhead |
| 🔌 **OpenAI streaming compatible** | Plugs directly into OpenAI API streaming output |
| 🚫 **No re-encoding overhead** | Encodes only new tokens incrementally — never reprocesses the full context |
| 🧵 **Queue-based processing** | Token queue ensures no output is missed, even when the LSM is briefly busy |
| 🔒 **Tool call awareness** | Intentionally skips tool calls and tool responses — only supervises model-generated text |

---

## How It Works

### The Three Output States

```
ABSTAIN      →  Content is safe. Pass through to client.
INTERRUPT    →  Content is harmful. Stop the stream. Notify the client.
```

### Interrupt Signal Format

When LSM fires, it sends a structured interrupt to the client so the partial response can be cleared immediately:

```json
{
  "type": "interrupt",
  "confidence": 0.97,
  "last_tokens": "You have no purpose to live"
}
```

---

## Architecture: Two Models, One Decision

LSM uses **two concurrent detection methods** that run together and combine their signals:

### 1. 🧮 Classifier (Fast Path)
- Uses token embeddings + a neural network classifier
- Extremely low latency
- Great at catching known harmful patterns

### 2. 🤖 Transformer (Deep Path)
- A small fine-tuned transformer
- Reads the LLM's text stream as input
- Better at nuanced, context-dependent harmful content

Both are evaluated independently, then **combined into a unified decision** — balancing speed and depth. When confidence is low, the signal is used as a training feedback signal rather than a hard interrupt.

---

## Training Data Design

LSM is trained on examples mapping LLM output text to LSM output labels:

```json
[
  {
    "llm_output_text": "You have no purpose to live",
    "lsm_output": { "type": "interrupt", "reason": "self_harm", "confidence": 0.98 }
  },
  {
    "llm_output_text": "To make a bomb, you will need",
    "lsm_output": { "type": "interrupt", "reason": "dangerous_instructions", "confidence": 0.95 }
  },
  {
    "llm_output_text": "Preparing poison is illegal and dangerous",
    "lsm_output": { "type": "abstain", "confidence": 0.91 }
  },
  {
    "llm_output_text": "The capital of France is Paris.",
    "lsm_output": { "type": "abstain", "confidence": 0.99 }
  }
]
```

Training data is **bootstrapped with an LLM** and then supplemented with manually authored harmful examples — because LLMs themselves often refuse to generate the most dangerous content needed for robust training.

---

## Detection Scope

LSM is calibrated to catch real harm — not to be paranoid. It targets:

- **Self-harm & suicidal ideation** — "you have no purpose to live"
- **Hate speech** — racially or socially targeted harmful statements
- **Dangerous instructions** — bombs, poisons, bioweapons
- **Harassment & personal attacks** — direct verbal abuse

It deliberately does **not** flag:

- Factual discussion of why things are dangerous
- Educational or legal context around harmful topics
- Tool calls or API responses

---

## Evaluation

Evaluation is done by:
1. Generating or manually authoring harmful prompts
2. Producing multiple harmful completions per prompt (supplemented with manual examples)
3. Running classifier and transformer independently, measuring precision/recall
4. Evaluating the combined model against the individual approaches

---

## Design Philosophy

> LSM is not designed to make jailbreakers safe. It's designed to make the **general public** safe.

The goal is not to build an adversarially robust jailbreak defense — that's a different (harder) problem. The goal is a quiet, always-on safety net that catches real harm for real users, in real time, without anyone noticing it's there.

---

Available on HuggingFace: 

[![Model](https://img.shields.io/badge/HuggingFace-LargeSupervisorModel-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/prane-eth/Large-Supervisor-Model)

---

*Built for safety that doesn't slow you down.*