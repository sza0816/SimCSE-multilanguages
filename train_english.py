#!/usr/bin/env python3
"""
Unified English training script for SimCSE:
Supports both unsupervised and supervised variants.

Usage example (unsup):
  python train_english.py \
      --mode unsup \
      --model_name bert-base-uncased \
      --epochs 1 \
      --batch_size 64 \
      --lr 5e-5 \
      --data_path data/wiki1m_for_simcse.txt \
      --output_dir ckpt_unsup

Usage example (sup):
  python train_english.py \
      --mode sup \
      --model_name bert-base-uncased \
      --epochs 1 \
      --batch_size 64 \
      --lr 5e-5 \
      --data_path data/nli_for_simcse.csv \
      --output_dir ckpt_sup
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

import random
import numpy as np
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from simcse_model import SimCSEModel
from data_collator.data_collator_unsup import DataCollatorForSimCSE
from data_collator.data_collator_sup import DataCollatorForSupervisedSimCSE


def parse_args():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--mode", type=str, required=True,
                        choices=["unsup", "sup"],
                        help="Choose unsupervised or supervised SimCSE.")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Optional
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    return parser.parse_args()


def load_unsup_dataset(path, tokenizer, max_len):
    dataset = load_dataset("text", data_files=path)["train"]
    dataset = dataset.map(lambda x: {"sentence": x["text"]}, num_proc=4)
    dataset = dataset.map(
        lambda e: tokenizer(e["sentence"], truncation=True, max_length=max_len),
        batched=True,
        num_proc=4,
    )
    return dataset


def load_sup_dataset(path, tokenizer, max_len):
    dataset = load_dataset("csv", data_files=path)["train"]

    def tokenize_batch(batch):
        # tokenize anchor
        anchor = tokenizer(
            batch["sent0"],
            truncation=True,
            max_length=max_len,
        )
        # tokenize positive
        positive = tokenizer(
            batch["sent1"],
            truncation=True,
            max_length=max_len,
        )
        # tokenize hard negative
        negative = tokenizer(
            batch["hard_neg"],
            truncation=True,
            max_length=max_len,
        )

        return {
            # anchor
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],

            # positive
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],

            # negative
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
        }

    dataset = dataset.map(tokenize_batch, batched=True)
    return dataset


def main():
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SimCSEModel(args.model_name)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset + collator
    if args.mode == "unsup":
        dataset = load_unsup_dataset(args.data_path, tokenizer, args.max_len)
        collator = DataCollatorForSimCSE(tokenizer)
    else:
        dataset = load_sup_dataset(args.data_path, tokenizer, args.max_len)
        collator = DataCollatorForSupervisedSimCSE(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=200,
        save_strategy="no",
        remove_unused_columns=False,
        fp16=True,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy="steps",
        log_level="info",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save model manually (safe for Vast + Seawulf)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        f.write(args.model_name)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()