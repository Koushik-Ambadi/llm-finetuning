# model_training/tokenizer_utils.py

from transformers import AutoTokenizer
from model_training.config import (
    model_path,
    prompt_prefix,
    max_seq_length,
    padding_type,
    truncation_side,
    padding_side
)
from model_training.debug_utils import save_checkpoint


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # Set pad_token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Tokenizer] pad_token not found. Using eos_token ({tokenizer.eos_token}) as pad_token.")

    # Set tokenizer sides from config
    tokenizer.truncation_side = truncation_side
    tokenizer.padding_side = padding_side

    print(f"[Tokenizer] Loaded tokenizer from {model_path}")
    print(f"[Tokenizer] Truncation side: {tokenizer.truncation_side}")
    print(f"[Tokenizer] Padding side: {tokenizer.padding_side}")
    print(f"[Tokenizer] Max sequence length: {max_seq_length}")

    return tokenizer


def tokenize_function(batch, tokenizer):
    prompt_strings = []

    for user_input, assistant_output in zip(batch["input"], batch["output"]):
        if not isinstance(user_input, str) or not isinstance(assistant_output, str):
            print(f"[Warning] Skipping invalid input/output pair: {user_input}, {assistant_output}")
            continue

        # Build chat messages as per tokenizer's template
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ]

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_strings.append(prompt)
        except Exception as e:
            print(f"[Warning] Failed to apply chat template. Input: {user_input[:30]}... Error: {e}")
            continue

    if not prompt_strings:
        raise ValueError("No valid prompt strings generated from chat template.")

    # Tokenize all prompts (batched)
    model_inputs = tokenizer(
        prompt_strings,
        padding="max_length",  # fixed size for training (required for batching)
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        return_tensors=None  # keep as dict of lists
    )

    # Use input_ids as labels
    model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]

    return model_inputs
