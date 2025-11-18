# data_collator/data_collator_sup.py
"""
This data collator prepares batches for supervised SimCSE using NLI triplets:
anchor, positive, and hard negative sentences. Each component is tokenized
separately to ensure clarity and correctness in the contrastive learning setup.
"""
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class DataCollatorForSupervisedSimCSE:
    tokenizer: AutoTokenizer
    max_length: int = 32

    # The examples argument contains triplet fields (anchor, positive, hard negative).
    # This method tokenizes each component separately to produce tensors used in the supervised InfoNCE objective.
    def __call__(self, examples):
        # Supervised SimCSE relies on NLI triplets: anchor, positive, and hard negative sentences.
        anchors = [ex["sent0"] for ex in examples]
        positives = [ex["sent1"] for ex in examples]
        hardnegs = [ex["hard_neg"] for ex in examples]

        # Tokenize each segment independently to avoid mixing attention masks or token_type_ids.
        anchor_batch = self.tokenizer(
            anchors,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        pos_batch = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg_batch = self.tokenizer(
            hardnegs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Return dict for SimCSE forward() supervised mode.
        # These keys are consumed by the model’s supervised forward pass to compute anchor–positive vs. hard-negative similarity.
        return {
            "input_ids": anchor_batch["input_ids"],
            "attention_mask": anchor_batch["attention_mask"],
            "pos_input_ids": pos_batch["input_ids"],
            "pos_attention_mask": pos_batch["attention_mask"],
            "neg_input_ids": neg_batch["input_ids"],
            "neg_attention_mask": neg_batch["attention_mask"],
        }