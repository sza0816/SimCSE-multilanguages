# evaluate.py
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize

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
    model.to(args.device)
    model.eval()

    # Load dataset
    df = pd.read_csv(args.test_file)

    sents1 = df["sentence1"].tolist()
    sents2 = df["sentence2"].tolist()
    scores = df["score"].tolist()

    print(f"Loaded {len(df)} sentence pairs.")

    # Encode both sides
    emb1 = encode(model, tokenizer, sents1, args.batch_size, args.max_len, args.device)
    emb2 = encode(model, tokenizer, sents2, args.batch_size, args.max_len, args.device)

    # Cosine similarity
    cos_sim = (emb1 * emb2).sum(axis=1)

    # Spearman correlation
    sp = spearmanr(cos_sim, scores).correlation
    print("\n==== Evaluation Result ====")
    print(f"Spearman correlation: {sp:.4f}")


if __name__ == "__main__":
    main()