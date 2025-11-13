import torch
import torch.nn as nn
from transformers import AutoModel

class SimCSEModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", pooling="cls", temperature=0.05):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.temperature = temperature

    def _pool(self, outputs, attention_mask):
        """Different pooling strategies"""
        if self.pooling == "cls":
            return outputs.last_hidden_state[:, 0]  # CLS
        elif self.pooling == "avg":
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            return torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Unsupervised SimCSE:
        Each input sentence is duplicated in the dataloader.
        Example batch shape:
            input_ids:  (batch, 2, seq_len)
        """
        # Unpack two views of the same sentence
        bsz = input_ids.shape[0]

        # Flatten (bsz, 2, seq_len) → (bsz*2, seq_len)
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self._pool(outputs, attention_mask)

        # (bsz*2, hidden)
        z1, z2 = embeddings[0::2], embeddings[1::2]

        # -------- SimCSE Contrastive Loss -------- #
        cos_sim = torch.matmul(z1, z2.t()) / self.temperature  # (bsz, bsz)
        labels = torch.arange(bsz).to(cos_sim.device)
        loss = nn.CrossEntropyLoss()(cos_sim, labels)

        return {"loss": loss, "embeddings": embeddings}