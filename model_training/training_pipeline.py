# model_training/training_pipeline.py

import os
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force training to use only GPU 0


from transformers import set_seed
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

from model_training.config import (
    csv_path, output_dir, train_batch_size,
    gradient_accumulation, num_epochs, learning_rate
)
from model_training.dataset_utils import load_dataset, preprocess_dataset
from model_training.tokenizer_utils import load_tokenizer, tokenize_function
from model_training.model_utils import load_model
from model_training.logging_utils import setup_main_logger, start_hardware_logging
#from model_training.data_collator import data_collator

#custom trainer
from trl import SFTTrainer
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)




def train_model():
    set_seed(42)

    # Setup logging
    main_log_path = os.path.join(output_dir, "training.log")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_main_logger(main_log_path, level=logging.INFO)
    logger.info("Starting training pipeline")

    # Start hardware logging
    stop_event, log_thread = start_hardware_logging(output_dir, interval=5)
    logger.info("Started hardware logging thread")

    try:
        dataset = load_dataset(csv_path)
        logger.info("Loaded dataset")
        split_dataset = preprocess_dataset(dataset)
        logger.info("Split dataset into train/test")

        tokenizer = load_tokenizer()
        logger.info("Tokenizer loaded")

        model = load_model()
        logger.info("Model loaded")

        tokenized_train = split_dataset["train"].map(
            lambda x: tokenize_function(x, tokenizer),
            remove_columns=[col for col in split_dataset["train"].column_names if col in ["input", "output", "requirement_description", "test_steps"]],
            batched=True
        )
        logger.info("Tokenized training set")

        tokenized_test = split_dataset["test"].map(
            lambda x: tokenize_function(x, tokenizer),
            remove_columns=[col for col in split_dataset["train"].column_names if col in ["input", "output", "requirement_description", "test_steps"]],
            batched=True
        )
        logger.info("Tokenized test set")

        tokenized_train.set_format("torch")
        tokenized_test.set_format("torch")
        logger.info("Set tensor format for datasets")

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
            dataset_text_field=None,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            use_liger_kernel=False
        )
        logger.info(f"Config: batch_size={train_batch_size}, grad_accum={gradient_accumulation}, epochs={num_epochs}, lr={learning_rate}")

        accelerator = Accelerator()
        logger.info("Accelerator initialized")
        device = accelerator.device  # use Accelerate's chosen device
        model = model.to(device)

        trainer = CustomSFTTrainer(
            model=model,
            args=config,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test
        )



        logger.info("Starting training")
        trainer.train()
        logger.info("Training finished")

    finally:
        logger.info("Stopping hardware logging")
        stop_event.set()
        log_thread.join()
        logger.info("Hardware logging stopped")

        logger.info("Saving model and tokenizer")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved model and tokenizer")



