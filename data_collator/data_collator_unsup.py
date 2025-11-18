"""
This data collator prepares batches for unsupervised SimCSE by duplicating each sentence 
to create two dropout-based views that will act as positive pairs for InfoNCE.
"""

class DataCollatorForSimCSE:
    def __init__(self, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length

    # features is a list of dataset samples; this function extracts sentences, duplicates them,
    # tokenizes them, and returns a batch suitable for SimCSE contrastive learning.
    def __call__(self, features):
        # Datasets may use different keys for sentences; this collator supports both.
        if "text" in features[0]:
            key = "text"
        elif "sentence" in features[0]:
            key = "sentence"
        else:
            raise KeyError("Dataset must contain either 'text' or 'sentence'")
        sentences = [f[key] for f in features]

        # Duplicating sentences creates positive pairs for unsupervised SimCSE.
        sentences = sentences + sentences

        # This tokenizes 2N sentences at once to feed the model.
        batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove token_type_ids because single-sentence input does not need them.
        if "token_type_ids" in batch:
            batch.pop("token_type_ids")

        return batch