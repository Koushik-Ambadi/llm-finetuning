# model_training/tokenizer_utils.py

from transformers import AutoTokenizer
from model_training.config import model_path

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_function(example, tokenizer):
    input_text = example["input"] or ""
    output_text = example["output"] or ""
    combined = input_text + "\n" + output_text

    tokenized = tokenizer(
        combined,
        max_length=2048,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors=None
    )

    labels = [
        token if token != tokenizer.pad_token_id else -100
        for token in tokenized["input_ids"]
    ]

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }
