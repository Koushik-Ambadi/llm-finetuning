# model_training/tokenizer_utils.py

from transformers import AutoTokenizer
from copy import deepcopy
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
    assistant_token_start_indices = []

    for user_input, assistant_output in zip(batch["input"], batch["output"]):
        if not isinstance(user_input, str) or not isinstance(assistant_output, str):
            print(f"[Warning] Skipping invalid input/output pair: {user_input}, {assistant_output}")
            continue

        # Chat-style message formatting
        messages = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output}
        ]

        try:
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_strings.append(full_prompt)

            # Tokenize without truncation to find split point
            tokenized_full = tokenizer(full_prompt, add_special_tokens=False)
            tokenized_user_only = tokenizer(
                tokenizer.apply_chat_template(messages[:1], tokenize=False),
                add_special_tokens=False
            )

            assistant_token_start = len(tokenized_user_only["input_ids"])
            assistant_token_start_indices.append(assistant_token_start)

        except Exception as e:
            print(f"[Warning] Failed to apply chat template. Input: {user_input[:30]}... Error: {e}")
            continue

    if not prompt_strings:
        raise ValueError("No valid prompt strings generated from chat template.")

    # Final tokenization with padding and truncation
    model_inputs = tokenizer(
        prompt_strings,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        return_tensors=None
    )

    # Prepare labels (mask non-assistant tokens and pad tokens)
    pad_token_id = tokenizer.pad_token_id
    model_inputs["labels"] = []

    for input_ids, start_idx in zip(model_inputs["input_ids"], assistant_token_start_indices):
        labels = deepcopy(input_ids)

        for i in range(len(labels)):
            if i < start_idx or labels[i] == pad_token_id:
                labels[i] = -100  # Mask

        model_inputs["labels"].append(labels)

    return model_inputs


"""

def tokenize_function(batch, tokenizer):
    prompt_strings = []

    for user_input, assistant_output in zip(batch["input"], batch["output"]):
        if not isinstance(user_input, str) or not isinstance(assistant_output, str):
            print(f"[Warning] Skipping invalid input/output pair: {user_input}, {assistant_output}")
            continue

        # Build chat messages using chat template
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
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        return_tensors=None
    )

    # Mask pad tokens with -100 in labels
    pad_token_id = tokenizer.pad_token_id
    model_inputs["labels"] = []

    for input_ids in model_inputs["input_ids"]:
        labels = deepcopy(input_ids)
        labels = [token_id if token_id != pad_token_id else -100 for token_id in labels]
        model_inputs["labels"].append(labels)

    return model_inputs

"""