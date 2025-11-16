class DataCollatorForSimCSE:
    def __init__(self, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # Extract list of sentences
        if "text" in features[0]:
            key = "text"
        elif "sentence" in features[0]:
            key = "sentence"
        else:
            raise KeyError("Dataset must contain either 'text' or 'sentence'")
        sentences = [f[key] for f in features]

        # Duplicate: two views for each sentence
        sentences = sentences + sentences

        # Tokenize
        batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove token_type_ids
        if "token_type_ids" in batch:
            batch.pop("token_type_ids")

        return batch