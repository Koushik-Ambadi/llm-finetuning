# model_training/training_pipeline.py

import os
import logging
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force training to use only GPU 0

from transformers import set_seed
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

from model_training.config import (
    csv_path, output_dir, train_batch_size,
    gradient_accumulation, num_epochs, learning_rate,token_batchsize
)
from model_training.dataset_utils import load_dataset, preprocess_dataset
from model_training.tokenizer_utils import load_tokenizer, tokenize_function
from model_training.model_utils import load_model
from model_training.logging_utils import setup_main_logger, start_hardware_logging
from model_training.debug_utils import save_checkpoint

# Optional custom trainer
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

def debug_log_tokenized_sample(logger, tokenized_dataset, tokenizer, num_samples=3):
    logger.info(f"\n📌 Logging first {num_samples} tokenized samples for debugging:")
    for i in range(min(len(tokenized_dataset), num_samples)):
        sample = tokenized_dataset[i]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

        logger.info(f"\n🔹 Sample {i + 1}")
        logger.info(f"Token count: {len(input_ids)}")
        logger.info(f"Decoded prompt:\n{decoded[:1000]}{'...' if len(decoded) > 1000 else ''}")
        logger.info(f"Attention mask (first 50): {attention_mask[:50]}{'...' if len(attention_mask) > 50 else ''}")
        num_zeros = (attention_mask == 0).sum().item()
        logger.info(f"Number of masked (0) tokens: {num_zeros}")


        logger.info("-" * 60)

def train_model():
    set_seed(42)

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_main_logger(os.path.join(output_dir, "training.log"), level=logging.INFO)
    logger.info("🚀 Starting training pipeline")

    # Start hardware logging
    stop_event, log_thread = start_hardware_logging(output_dir, interval=5)
    logger.info("🧠 Started hardware logging")

    try:
        dataset = load_dataset(csv_path)
        logger.info("✅ Loaded dataset")

        split_dataset = preprocess_dataset(dataset)
        logger.info("✅ Split dataset into train/test")

        tokenizer = load_tokenizer()
        logger.info("✅ Tokenizer loaded")
        logger.info(f"Tokenizer class: {tokenizer.__class__}")
        logger.info(f"Pad token: {tokenizer.pad_token} | EOS token: {tokenizer.eos_token}")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

        model = load_model()
        logger.info("✅ Model loaded")

        # Tokenize training and test datasets
        logger.info("🔄 Tokenizing training set...")
        tokenized_train = split_dataset["train"].map(
            partial(tokenize_function, tokenizer=tokenizer),
            remove_columns=["input", "output", "requirement_description", "test_steps"],
            batched=True,
            #batch_size=token_batchsize,
        )

        logger.info("🔄 Tokenizing test set...")
        tokenized_test = split_dataset["test"].map(
            partial(tokenize_function, tokenizer=tokenizer),
            remove_columns=["input", "output", "requirement_description", "test_steps"],
            batched=True,
            #batch_size=token_batchsize,
        )

        tokenized_train.set_format("torch")
        tokenized_test.set_format("torch")

        logger.info("✅ Tokenization complete")

        # 🔍 DEBUG: Log tokenized samples and attention masks
        debug_log_tokenized_sample(logger, tokenized_train, tokenizer, num_samples=3)

        # Config
        config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=num_epochs,
            logging_steps=5,
            learning_rate=learning_rate,
            save_steps=100,
            save_total_limit=3,
            report_to="none",
            dataset_text_field=None,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            use_liger_kernel=False,
        )
        logger.info(f"🛠 Config: batch_size={train_batch_size}, grad_accum={gradient_accumulation}, epochs={num_epochs}, lr={learning_rate}")

        # Accelerator
        accelerator = Accelerator()
        logger.info("✅ Accelerator initialized")
        model = model.to(accelerator.device)

        # Trainer
        trainer = CustomSFTTrainer(
            model=model,
            args=config,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test
        )


        # After initializing the trainer, decide whether to resume or not
        checkpoint_dir = output_dir
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]

        if checkpoints:
            # There are checkpoint folders, find the latest checkpoint
            latest_checkpoint = max(
                checkpoints,
                key=lambda x: int(x.split("-")[-1])
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            # No checkpoint found, start fresh
            logger.info("No checkpoint found, starting training from scratch")
            trainer.train()
    finally:
        logger.info("🛑 Stopping hardware logging")
        stop_event.set()
        log_thread.join()
        logger.info("✅ Hardware logging stopped")

        logger.info("💾 Saving model and tokenizer")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("✅ Model and tokenizer saved")
