# evaluate.py
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize
import os


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def detect_columns(df):
    col_s1 = None
    col_s2 = None
    col_score = None
    for c in df.columns:
        lc = c.lower()
        if col_s1 is None and ("sentence1" in lc or "sent1" in lc or "s1" == lc or "seq1" in lc):
            col_s1 = c
        if col_s2 is None and ("sentence2" in lc or "sent2" in lc or "s2" == lc or "seq2" in lc):
            col_s2 = c
        if col_score is None and ("score" in lc or "label" in lc or "gold" in lc or "gold_score" in lc or "similarity" in lc):
            col_score = c
    if col_s1 is None or col_s2 is None or col_score is None:
        raise ValueError("Could not auto-detect sentence1/sentence2/score columns in test file.")
    return col_s1, col_s2, col_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def encode(model, tokenizer, sentences, batch_size=64, max_len=32, device="cuda"):
    """Encode sentences into embeddings."""
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i: i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            if device.startswith("cuda"):
                try:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                except Exception:
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            # SimCSE representation = CLS hidden state
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def main():
    args = parse_args()

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = "cpu"

    model.to(device)
    model.eval()

    # Load dataset with automatic separator detection
    df = pd.read_csv(args.test_file, sep="\t")
    if len(df.columns) < 3:
        df = pd.read_csv(args.test_file, sep=",")

    col_s1, col_s2, col_score = detect_columns(df)

    # Drop rows where score column cannot be converted to float (skip headers or invalid rows)
    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False
    df = df[df[col_score].apply(is_float)]

    sents1 = df[col_s1].astype(str).tolist()
    sents2 = df[col_s2].astype(str).tolist()
    scores = df[col_score].astype(float).tolist()

    print(f"Loaded {len(df)} sentence pairs from file: {args.test_file}")

    # Encode both sides
    emb1 = encode(model, tokenizer, sents1, args.batch_size, args.max_len, device)
    emb2 = encode(model, tokenizer, sents2, args.batch_size, args.max_len, device)

    # Cosine similarity
    cos_sim = (emb1 * emb2).sum(axis=1)

    # Spearman correlation
    sp = spearmanr(cos_sim, scores).correlation

    print("\n--- Evaluation Result ---")
    print(f"File: {args.test_file}")
    print(f"Model: {args.model_path}")
    print(f"Number of sentence pairs: {len(df)}")
    print(f"Spearman: {sp:.4f}",flush = True)

    # Write summary into the model's evaluation folder
    eval_dir = os.path.join(args.model_path, "../eval")
    eval_dir = os.path.abspath(eval_dir)

    os.makedirs(eval_dir, exist_ok=True)
    summary_path = os.path.join(eval_dir, "summary.txt")
    with open(summary_path, "a") as f:
        f.write(f"{os.path.basename(args.test_file)}\t{sp:.4f}\n")


if __name__ == "__main__":
    main()