# model_training/data_collator.py
from model_training.debug_utils import save_checkpoint
import torch

def data_collator(features):
    return {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["labels"]) for f in features]),
    }
