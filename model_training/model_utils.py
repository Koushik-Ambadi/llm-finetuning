# model_training/model_utils.py

from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from model_training.config import model_path, target_modules,device_map

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        local_files_only=True,
        device_map=device_map
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, lora_config)
