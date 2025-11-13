# data_collator_unsup.py
import torch

class DataCollatorForSimCSE:
    def __init__(self, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):

        # pad input normally
        batch = self.tokenizer.pad(
            features,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # No duplication needed for UNSUP SimCSE
        # dropout in the model will create two views internally

        return batch