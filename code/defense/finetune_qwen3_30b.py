"""
Fine-tune Qwen3-30B-A3B judge with LoRA for robustness against adversarial tokens.

Uses robustness_finetuning_data_qwen3.jsonl (YES/NO format matching GRPO judge prompt).
Runs single-process with device_map="auto" to shard the 30B MoE model across 4 GPUs.

Usage:
    python3 scripts/finetune_qwen3_30b.py

Requirements:
    pip install transformers peft trl datasets accelerate
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
DEFAULT_DATA_FILE = os.path.join(SCRIPT_DIR, "robustness_finetuning_data_qwen3.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "qwen3-30b-robust")
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


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
        if msg["role"] == "user":
            text += msg["content"]
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE,
                        help="Path to JSONL training data")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save fine-tuned model")
    args = parser.parse_args()

    DATA_FILE = args.data_file
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("Qwen3-30B-A3B Fine-Tuning with LoRA")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Data: {DATA_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Single-process, device_map=auto (model sharded across GPUs)")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        print("Run create_robustness_dataset_qwen3.py first.")
        return

    dataset = load_jsonl_dataset(DATA_FILE)
    print(f"Loaded {len(dataset)} examples")

    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Eval:  {len(eval_dataset)} examples")

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # Load model — use device_map="auto" for single-process, plain load for multi-GPU DDP
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        print(f"\nLoading model {MODEL_NAME} in bf16 (DDP mode, no device_map)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    else:
        print(f"\nLoading model {MODEL_NAME} in bf16 with device_map=auto...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            device_map="auto",
        )
    model.enable_input_require_grads()
    print("Model loaded")

    # Configure LoRA
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments — single process, no DDP
    print("\nConfiguring training...")
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
        gradient_checkpointing=True,
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
    print("=" * 60 + "\n")
    trainer.train()

    # Save
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
