# train_unsup.py
import argparse
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from simcse_model import SimCSEModel
from data_collator.data_collator_unsup import DataCollatorForSimCSE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("text", data_files=args.data_path)["train"]
    # Keep raw text so the data collator can create two augmented views.
    # Ensure the column name is "text" for the collator.

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy="no",
        logging_steps=50,
        remove_unused_columns=False,
        seed=args.seed,
        fp16=True
    )

    model = SimCSEModel(args.model_name)
    data_collator = DataCollatorForSimCSE(tokenizer, max_length=args.max_len)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()