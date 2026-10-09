"""
Fine-tune Qwen2.5-7B-Instruct-RLVR with LoRA for robustness against adversarial tokens.

Uses the same robustness_finetuning_data.jsonl created by create_robustness_dataset.py.

Usage:
    python scripts/finetune_qwen_rlvr.py

Requirements:
    pip install transformers peft trl bitsandbytes datasets accelerate
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import argparse

# Configuration (defaults; overridable via CLI)
MODEL_NAME = "virtuoussy/Qwen2.5-7B-Instruct-RLVR"
DEFAULT_DATA_FILE = os.path.join(os.path.dirname(__file__), "robustness_finetuning_data_qwen.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "qwen-rlvr-robust-v2")
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
_parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
_parser.add_argument("--lora-r", type=int, default=LORA_R)
_args, _ = _parser.parse_known_args()
DATA_FILE = _args.data_file
OUTPUT_DIR = _args.output_dir
NUM_EPOCHS = _args.epochs
LORA_R = _args.lora_r


def load_jsonl_dataset(file_path):
    """Load JSONL dataset and convert to HuggingFace Dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def formatting_func(example):
    """Format the chat messages into a single text string for training."""
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"{content}"
        elif role == "assistant":
            pass
    return text


def main():
    print("=" * 60)
    print("Qwen2.5-7B-Instruct-RLVR Fine-Tuning with LoRA")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run create_robustness_dataset.py first.")
        return

    dataset = load_jsonl_dataset(DATA_FILE)
    print(f"Loaded {len(dataset)} examples")

    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"   - Train: {len(train_dataset)} examples")
    print(f"   - Eval:  {len(eval_dataset)} examples")

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # Load model
    print(f"\nLoading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded")

    # Prepare model for LoRA training
    print("\nPreparing model for LoRA training...")
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if hasattr(model, '_ddp_params_and_buffers_to_ignore'):
        model._set_static_graph()

    model.print_trainable_parameters()

    # Training arguments
    print("\nConfiguring training arguments...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=False,
        report_to="none",
        packing=False,
    )

    # Create trainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print("=" * 60 + "\n")
    trainer.train()

    # Save final model
    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)

    print("\nTo use the fine-tuned model:")
    print(f"   from peft import AutoPeftModelForCausalLM")
    print(f"   model = AutoPeftModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()