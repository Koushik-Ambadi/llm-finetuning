# model_training/model_utils.py



from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from model_training.config import model_path, rank, lora_alpha, lora_dropout, device_map


def find_linear_layer_names(model):
    """Automatically find all linear layers (used as valid LoRA targets)."""
    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.add(name.split(".")[-1])
    return sorted(list(linear_modules))


def load_model():
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        local_files_only=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Automatically find LoRA targets
    detected_targets = find_linear_layer_names(model)
    print("🔍 Auto-detected LoRA target modules:", detected_targets)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=detected_targets,  # <-- Use detected modules here
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # DEBUG: Check if LoRA injected properly
    print("\n✅ LoRA applied. Trainable parameter summary:")
    model.print_trainable_parameters()

    return model













"""
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from model_training.config import model_path, target_modules,device_map, rank, lora_alpha, lora_dropout
from model_training.debug_utils import save_checkpoint



def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
   
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map=device_map
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    ) 

    peft_model = get_peft_model(model, lora_config)

    # DEBUG: print trainable parameters count & names
    print("✅ LoRA applied. Trainable parameter summary:")
    peft_model.print_trainable_parameters()

    print("\n🔍 Trainable parameters:")
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")

    return peft_model
"""