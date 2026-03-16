# LLM Fine-Tuning

A complete, beginner-friendly tutorial for fine-tuning **Mistral 7B Instruct v0.3** using QLoRA on Google Colab's free tier (T4 GPU).

## What's Inside

| File | Description |
|------|-------------|
| `finetune-llm-tutorial.md` | Step-by-step written guide (in French) — from account setup to inference |
| `finetuning.py` | Full training script (Colab notebook export) |
| `data.jsonl`, `batch_*.jsonl` | Training data — curated Twitter/X thread examples |

## Stack

- **Transformers** + **TRL** — training loop with SFTTrainer
- **PEFT** + **BitsAndBytes** — 4-bit QLoRA for memory-efficient fine-tuning
- **Hugging Face Hub** — model download and upload

## Why Mistral 7B?

- **Sweet spot** for 500 examples: enough capacity for creative style transfer, small enough for a free T4 16GB
- **Instruct version**: already instruction-following — fine-tuning only teaches *style*, not task comprehension
- Smaller models (1.5B–3B) produce flat output; larger models (12B+) underfit with this dataset size

## Quick Start

1. Open `finetuning.py` in [Google Colab](https://colab.research.google.com)
2. Enable GPU runtime (T4)
3. Set your `HF_TOKEN` as environment variable
4. Run all cells (~1-2h training time)

## Purpose

End-to-end tutorial demonstrating LLM fine-tuning for style transfer — adapting a foundation model to generate content in a specific writing style.
