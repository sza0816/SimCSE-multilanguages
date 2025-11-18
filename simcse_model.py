"""
This file implements the core SimCSE model, supporting both unsupervised (dropout-based) and supervised (NLI triplet-based) training.
Both branches use an InfoNCE-style contrastive loss to learn sentence embeddings.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class SimCSEModel(nn.Module):
    def __init__(self, pretrained_model, temperature=0.05, supervised=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.temperature = temperature
        self.supervised = supervised

    # Encode sentences using the pretrained backbone and return a normalized [CLS] embedding.
    def encode(self, input_ids, attention_mask):
        """Encode sentence and return normalized CLS vector."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        return nn.functional.normalize(cls, dim=-1)

    def forward(self,
                input_ids=None, attention_mask=None,
                pos_input_ids=None, pos_attention_mask=None,
                neg_input_ids=None, neg_attention_mask=None):

        # ---------------------------------------------------------------------------------
        # Unsupervised SimCSE mechanism:
        # - Generate two dropout views of the same input sentence to create positive pairs.
        # - Concatenate these embeddings to form a batch of size 2N.
        # - Compute cosine similarity matrix scaled by temperature.
        # - Mask the diagonal to exclude self-similarity.
        # - Construct labels indicating positive pairs are offset by N in the batch.
        # - Apply InfoNCE loss via CrossEntropyLoss to maximize agreement between positive pairs.
        # ---------------------------------------------------------------------------------
        if not self.supervised:
            # two dropout views
            z1 = self.encode(input_ids, attention_mask)  # first view with dropout
            z2 = self.encode(input_ids, attention_mask)  # second view with dropout

            z = torch.cat([z1, z2], dim=0)  # (2N, hidden), concatenated embeddings

            # compute cosine similarity matrix and scale by temperature
            sim = torch.matmul(z, z.T) / self.temperature

            N = z1.size(0)

            # labels: for each example in z, the positive example is offset by N
            labels = torch.arange(N, device=z.device)
            labels = torch.cat([labels + N, labels], dim=0)

            # mask self-similarity on diagonal
            mask = torch.eye(2 * N, device=z.device).bool()
            sim = sim.masked_fill(mask, -1e4)

            # compute cross entropy loss with InfoNCE objective
            loss = nn.CrossEntropyLoss()(sim, labels)
            return {"loss": loss}

        # ---------------------------------------------------------------------------------
        # Supervised SimCSE (as in paper):
        # - Encode anchor, positive, and optionally hard negative sentences.
        # - Concatenate positive and negative embeddings.
        # - Labels are all zeros since positives are always at index 0.
        # - Compute similarity matrix scaled by temperature.
        # - Apply InfoNCE-style cross-entropy loss to learn to distinguish positives from negatives.
        # ---------------------------------------------------------------------------------
        # anchor: (N, hidden)
        anchor = self.encode(input_ids, attention_mask)              # encode anchor sentences
        positive = self.encode(pos_input_ids, pos_attention_mask)     # encode positive sentences

        if neg_input_ids is not None:
            # hard negatives
            negative = self.encode(neg_input_ids, neg_attention_mask) # encode hard negative sentences
            # concat [pos; neg] → shape (N+N_neg, hidden)
            embeddings = torch.cat([positive, negative], dim=0)
            # label is always the index of the positive (0)
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        else:
            embeddings = positive
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        # cosine similarity
        sim = torch.matmul(anchor, embeddings.T) / self.temperature
        loss = nn.CrossEntropyLoss()(sim, labels)

        return {"loss": loss}