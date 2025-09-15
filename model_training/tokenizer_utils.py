# model_training/tokenizer_utils.py
from transformers import AutoTokenizer
from model_training.config import model_path, prompt_prefix, max_seq_length, padding_type, truncation_side, padding_side

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # If pad_token not defined, set pad_token = eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set truncation/padding side from config
    tokenizer.truncation_side = truncation_side
    tokenizer.padding_side = padding_side  # or you could also get this from config

    return tokenizer

def tokenize_function(batch, tokenizer):
    # Build input texts with prompt_prefix
    inputs = [prompt_prefix + desc for desc in batch.get("requirement_description", [])]
    outputs = batch.get("test_steps", [])

    # Combine input + output
    combined = [inp + "\n" + out for inp, out in zip(inputs, outputs)]

    tokenized = tokenizer(
        combined,
        max_length=max_seq_length,
        padding=padding_type,        # True -> dynamic or based on collator
        truncation=True,
        return_attention_mask=True,
        #return_tensors="pt"
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Mask the labels: only output part should be used for loss
    # We need to find where input ends in each sequence
    # One simple way: tokenize the input_text (without output), count its length + special tokens
    input_tokenized = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding=padding_type,
        truncation=True,
        return_attention_mask=False,
        #return_tensors="pt"
    )["input_ids"]

    labels = [ids.copy() for ids in input_ids]  # deep copy each list

    for i in range(len(labels)):
        in_len = len(input_tokenized[i])
        labels[i][:in_len] = [-100] * in_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
