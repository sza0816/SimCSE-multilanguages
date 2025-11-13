# data_collator_unsup.py
import torch

class DataCollatorForSimCSE:
    def __init__(self, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):

        # normal batch
        batch = self.tokenizer.pad(
            features,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # IMPORTANT: deep copy tensors
        batch_dup = {
            "input_ids": batch["input_ids"].clone(),
            "attention_mask": batch["attention_mask"].clone()
        }

        # rename for model forward
        batch["input_ids_dup"] = batch_dup["input_ids"]
        batch["attention_mask_dup"] = batch_dup["attention_mask"]

        return batch