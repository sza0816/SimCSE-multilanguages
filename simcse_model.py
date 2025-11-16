import torch
import torch.nn as nn
from transformers import AutoModel


class SimCSEModel(nn.Module):
    def __init__(self, pretrained_model, temperature=0.05, supervised=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.temperature = temperature
        self.supervised = supervised

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

        # -------------------------------UNSUPERVISED SIMCSE-------------------------------
        if not self.supervised:
            # two dropout views
            z1 = self.encode(input_ids, attention_mask)
            z2 = self.encode(input_ids, attention_mask)

            z = torch.cat([z1, z2], dim=0)  # (2N, hidden)
            sim = torch.matmul(z, z.T) / self.temperature

            N = z1.size(0)

            # labels: [N..2N-1, 0..N-1]
            labels = torch.arange(N, device=z.device)
            labels = torch.cat([labels + N, labels], dim=0)

            # mask self
            mask = torch.eye(2 * N, device=z.device).bool()
            sim = sim.masked_fill(mask, -1e4)

            loss = nn.CrossEntropyLoss()(sim, labels)
            return {"loss": loss}

        # -------------------------------SUPERVISED SIMCSE-------------------------------
        anchor = self.encode(input_ids, attention_mask)
        positive = self.encode(pos_input_ids, pos_attention_mask)

        if neg_input_ids is not None:
            negative = self.encode(neg_input_ids, neg_attention_mask)
            embeddings = torch.cat([positive, negative], dim=0)
        else:
            embeddings = positive

        sim = torch.matmul(anchor, embeddings.T) / self.temperature
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=sim.device)

        loss = nn.CrossEntropyLoss()(sim, labels)
        return {"loss": loss}