from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from simcse import SimCSEModel
import torch

# testing GPU availability
print("GPU available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("csv", data_files="nli_for_simcse.csv")["train"]

    training_args = TrainingArguments(
        output_dir="./ckpt",
        per_device_train_batch_size=64,
        learning_rate=5e-5,
        num_train_epochs=0.01,          # 0.01 for testing
        fp16=False,
        logging_steps=50,
        save_strategy="no",
    )

    model = SimCSEModel("bert-base-uncased")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()