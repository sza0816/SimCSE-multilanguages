#!/usr/bin/env python3
"""
Evaluate English SimCSE (supports both unsupervised and supervised checkpoints).

This script:
    - Loads a saved checkpoint directory
    - Loads its backbone model (from config.txt)
    - Loads STS-B English evaluation data
    - Computes Spearman correlation for:
        * STS-B dev
        * STS-B test

Usage:
    python evaluate_english.py --ckpt_dir ckpt_unsup
    python evaluate_english.py --ckpt_dir ckpt_sup
"""

# can be used to evaluate both sup & unsup versions

import argparse
import os
import torch
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import pandas as pd

from transformers import AutoTokenizer
from simcse_model import SimCSEModel


#  ---------------------Compute cosine similarity---------------------
def cosine_sim(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


#  ---------------------Load STS-B (dev + test) from local---------------------
def load_stsb():
    dev = pd.read_csv("data/english/STS-B/original/sts-dev.tsv", sep="\t")
    test = pd.read_csv("data/english/STS-B/original/sts-test.tsv", sep="\t")
    return dev, test


# ---------------------Encode sentences using model---------------------
def encode(model, tokenizer, sentences, device):
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), 64)):
            batch = sentences[i : i + 64]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=32, return_tensors="pt").to(device)

            emb = model.encode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


#  ---------------------Evaluate a checkpoint---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory that contains pytorch_model.bin + config.txt")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir

    # Load backbone name
    config_path = os.path.join(ckpt_dir, "config.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.txt not found — cannot determine backbone model.")

    with open(config_path, "r") as f:
        backbone = f.read().strip()

    print(f"Loading backbone model: {backbone}")

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = SimCSEModel(backbone)

    # Load weights
    state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                            map_location="cpu")
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Device:", device)

    # Load STS-B data
    dev, test = load_stsb()

    # Encode sentences
    for name, df in [("dev", dev), ("test", test)]:
        print(f"\nEvaluating STS-B {name} ...")

        sents1 = df["sentence1"].tolist()
        sents2 = df["sentence2"].tolist()
        gold_scores = df["score"].astype(float).tolist()

        emb1 = encode(model, tokenizer, sents1, device)
        emb2 = encode(model, tokenizer, sents2, device)

        sims = cosine_sim(emb1, emb2).numpy()

        spearman = spearmanr(sims, gold_scores).correlation
        print(f"{name} Spearman: {spearman:.4f}")


if __name__ == "__main__":
    main()