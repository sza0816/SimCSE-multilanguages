#!/usr/bin/env python3

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
def cosine_sim(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


# ---------------------Encode sentences using model---------------------
def encode(model, tokenizer, sentences, device, batch_size, max_len):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i : i + batch_size]
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing pytorch_model.bin + config.txt")
    parser.add_argument("--test_file", type=str, required=True,
                        help="Single STS-B TSV file to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=32)
    args = parser.parse_args()

    ckpt_dir = args.model_path

    # Load backbone name
    config_path = os.path.join(ckpt_dir, "config.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.txt not found in checkpoint directory.")

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
    gold_scores = df["label"].astype(float).tolist()

    # Encode
    emb1 = encode(model, tokenizer, s1, device, args.batch_size, args.max_len)
    emb2 = encode(model, tokenizer, s2, device, args.batch_size, args.max_len)

    # Compute Spearman
    sims = cosine_sim(emb1, emb2).numpy()
    spearman = spearmanr(sims, gold_scores).correlation

    print(f"Spearman: {spearman:.4f}")


if __name__ == "__main__":
    main()