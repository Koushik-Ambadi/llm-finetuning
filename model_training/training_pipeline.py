# model_training/training_pipeline.py

from transformers import set_seed
from trl import SFTTrainer, SFTConfig

from model_training.config import (
    csv_path, output_dir, train_batch_size,
    gradient_accumulation, num_epochs, learning_rate
)
from model_training.dataset_utils import load_dataset, preprocess_dataset
from model_training.tokenizer_utils import load_tokenizer, tokenize_function
from model_training.data_collator import data_collator
from model_training.model_utils import load_model

def train_model():
    set_seed(42)

    dataset = load_dataset(csv_path)
    split_dataset = preprocess_dataset(dataset)

    tokenizer = load_tokenizer()
    model = load_model()

    tokenized_train = split_dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=["input", "output", "requirement_description", "test_steps"]
    )

    tokenized_test = split_dataset["test"].map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=["input", "output", "requirement_description", "test_steps"]
    )

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=num_epochs,
        logging_steps=10,
        learning_rate=learning_rate,
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        dataset_text_field=None
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator
    )

    trainer.tokenizer = tokenizer
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
