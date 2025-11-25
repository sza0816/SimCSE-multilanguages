#!/usr/bin/env python3

"""
This script evaluates a trained SimCSE checkpoint on STS-B files.
It computes embeddings for sentence pairs, calculates their cosine similarity,
and reports the Spearman correlation with the gold labels.
Supports evaluation for multiple languages.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from transformers import AutoTokenizer
from simcse_model import SimCSEModel


# ---------------------Compute cosine similarity---------------------
# Normalize embeddings and compute cosine similarity for each pair
def cosine_sim(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


# ---------------------Encode sentences using model---------------------
# Encode a list of sentences batch-wise using the model's encode() method and return stacked embeddings
def encode(model, tokenizer, sentences, device, batch_size, max_len):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i : i + batch_size]
            # Pad/truncate inputs and move to device
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            emb = model.encode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)


# ---------------------Evaluate checkpoint on a single STS-B file---------------------
# Load the trained model, read STS-B file, encode sentence pairs, compute similarity scores, and output Spearman correlation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing pytorch_model.bin + config.txt")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Single STS-B TSV file to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--lang", type=str, default="en", choices=["en","ch","hi"],
                        help="Language tag for bookkeeping; not used for auto path selection yet.")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Optional backbone name to override config.txt (e.g., bert-base-chinese).")
    args = parser.parse_args()

    ckpt_dir = args.model_path

    # Load backbone name
    config_path = os.path.join(ckpt_dir, "config.txt")
    if args.backbone is not None:
        backbone = args.backbone
    else:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "config.txt not found in checkpoint directory. "
                "Please pass --backbone to specify the pretrained model name."
            )
        with open(config_path, "r") as f:
            backbone = f.read().strip()

    print(f"Loading backbone model: {backbone}")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = SimCSEModel(backbone)

    # Load model weights
    state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                            map_location="cpu")
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Device:", device)

    # Load evaluation file
    df = pd.read_csv(args.test_file, sep="\t")
    s1 = df["sentence1"].tolist()
    s2 = df["sentence2"].tolist()

    # Automatic label column detection
    possible_labels = ["label", "score", "relatedness_score", "similarity"]
    found_label = None
    for col in possible_labels:
        if col in df.columns:
            found_label = col
            break

    if found_label is None:
        raise ValueError(
            f"No valid label column found. Expected one of {possible_labels}, but got {df.columns.tolist()}"
        )

    gold_scores = df[found_label].astype(float).tolist()
    print(f"Using label column: {found_label}")

    # Encode
    emb1 = encode(model, tokenizer, s1, device, args.batch_size, args.max_len)
    emb2 = encode(model, tokenizer, s2, device, args.batch_size, args.max_len)

    # Compute Spearman
    sims = cosine_sim(emb1, emb2).numpy()
    spearman = spearmanr(sims, gold_scores).correlation

    print(f"Spearman: {spearman:.4f}")


if __name__ == "__main__":
    main()