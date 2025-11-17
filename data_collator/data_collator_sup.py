# data_collator/data_collator_sup.py
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class DataCollatorForSupervisedSimCSE:
    tokenizer: AutoTokenizer
    max_length: int = 32

    def __call__(self, examples):
        # Extract sentences
        anchors = [ex["sent0"] for ex in examples]
        positives = [ex["sent1"] for ex in examples]
        hardnegs = [ex["hard_neg"] for ex in examples]

        # Tokenize separately so model receives clean tensors
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

        # Return dict for SimCSE forward() supervised mode
        return {
            "input_ids": anchor_batch["input_ids"],
            "attention_mask": anchor_batch["attention_mask"],
            "pos_input_ids": pos_batch["input_ids"],
            "pos_attention_mask": pos_batch["attention_mask"],
            "neg_input_ids": neg_batch["input_ids"],
            "neg_attention_mask": neg_batch["attention_mask"],
        }