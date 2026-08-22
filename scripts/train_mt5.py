import os
import sys
import json
import argparse
import logging
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import Dataset

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mycner.evaluation.metrics import evaluate_predictions

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

class ProductionLoggingCallback(TrainerCallback):
    """
    A custom callback to print training and evaluation metrics beautifully
    at the end of each epoch to ensure clear and actionable feedback.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"--- Epoch {state.epoch:.1f} / {args.num_train_epochs} Finished ---")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                logger.info(f"Step {state.global_step} | Training Loss: {logs['loss']:.4f}")
            if "eval_loss" in logs:
                logger.info(f"--- Validation Metrics (Step {state.global_step}) ---")
                logger.info(f"  Eval Loss: {logs['eval_loss']:.4f}")
                for key in ["eval_precision", "eval_recall", "eval_f1", "eval_exact_match_count", "eval_invalid_generated_count"]:
                    if key in logs:
                        logger.info(f"  {key.replace('eval_', '').title()}: {logs[key]}")
                logger.info("--------------------------------------------")

def read_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def parse_args():
    parser = argparse.ArgumentParser(description="Production-level fine-tuning pipeline for mT5 Agricultural CNER.")
    
    # Model & Tokenizer settings
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="experiments/mt5", help="Output directory.")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (auto-configured if not specified).")
    parser.add_argument("--train_batch_size", type=int, default=None, help="Train batch size (auto-configured if not specified).")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Eval batch size (auto-configured if not specified).")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_input_length", type=int, default=128, help="Max input sequence length.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max target sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # Logging & Strategy
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Where to report metrics ('tensorboard', 'wandb', 'none').")
    parser.add_argument("--logging_steps", type=int, default=None, help="Logging frequency.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    logger.info("Initializing Fine-Tuning Pipeline...")
    
    # Check GPU availability and configure production-level auto-tuning
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    logger.info(f"Training Device: {device_name}")
    
    # Auto-configure memory configurations based on model size
    is_base_model = "base" in args.model_name.lower()
    
    if cuda_available:
        logger.info("🚀 CUDA GPU detected! Optimizing with FP16 and high-performance hyperparameters.")
        if is_base_model:
            logger.info("📦 Detecting 'base' size model. Applying memory-efficient configurations (Batch Size: 8, Accumulation: 2, Checkpointing: True) to prevent OOM on T4 GPUs.")
            train_batch_size = args.train_batch_size or 8
            eval_batch_size = args.eval_batch_size or 8
            gradient_accumulation_steps = args.gradient_accumulation_steps or 2
            use_gradient_checkpointing = True
            # For mt5-base, a learning rate of 1e-4 is much more stable than 3e-4 to prevent NaN/divergence
            learning_rate = args.learning_rate if args.learning_rate != 3e-4 else 1e-4
        else:
            logger.info("⚡ Detecting 'small' size model. Applying standard fast configuration (Batch Size: 16).")
            train_batch_size = args.train_batch_size or 16
            eval_batch_size = args.eval_batch_size or 16
            gradient_accumulation_steps = args.gradient_accumulation_steps or 1
            use_gradient_checkpointing = False
            learning_rate = args.learning_rate
        use_fp16 = True
        epochs = args.epochs or 5
        logging_steps = args.logging_steps or 50
    else:
        logger.info("⚠️ Running on CPU. Downscaling batch sizes and epochs to prevent overhead.")
        train_batch_size = args.train_batch_size or 4
        eval_batch_size = args.eval_batch_size or 4
        gradient_accumulation_steps = args.gradient_accumulation_steps or 1
        use_gradient_checkpointing = False
        learning_rate = args.learning_rate
        use_fp16 = False
        epochs = args.epochs or 1
        logging_steps = args.logging_steps or 10

    # Read data
    logger.info("Loading train, validation and test datasets...")
    train_records = read_jsonl("data/processed/train.jsonl")
    val_records = read_jsonl("data/processed/validation.jsonl")
    test_records = read_jsonl("data/processed/test.jsonl")
    
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info(f"Loading pretrained model: {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # HF Datasets creation
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)
    test_dataset = Dataset.from_list(test_records)
    
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], 
            max_length=args.max_input_length, 
            truncation=True
        )
        labels_tokens = tokenizer(
            text_target=examples["target_text"], 
            max_length=args.max_target_length, 
            truncation=True
        )
        model_inputs["labels"] = labels_tokens["input_ids"]
        return model_inputs
        
    logger.info("Preprocessing datasets with Hugging Face map...")
    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
    val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
    test_tokenized = test_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
    
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Calculate warmup_steps as warmup_ratio is not supported in this transformers version
    num_samples = len(train_records)
    steps_per_epoch = max(1, num_samples // train_batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Calculated warmup_steps: {warmup_steps} (based on {total_steps} total steps with warmup_ratio of {args.warmup_ratio})")

    # Configure production training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Optimize on Entity-Level F1 for best performance
        greater_is_better=True,
        fp16=use_fp16,
        use_cpu=not cuda_available,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=use_gradient_checkpointing,
        report_to=args.report_to,
        dataloader_num_workers=2 if cuda_available else 0,
        disable_tqdm=False,
        optim="adamw_torch",  # Use standard, extremely stable AdamW for fine-tuning
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Compute metrics for Seq2Seq CNER
    def compute_metrics(eval_preds):
        preds, labels_ids = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        import numpy as np
        # Safeguard bounds
        preds = np.where((preds >= 0) & (preds < len(tokenizer)), preds, tokenizer.pad_token_id)
        labels_ids = np.where((labels_ids >= 0) & (labels_ids < len(tokenizer)), labels_ids, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
        
        # Post-process
        decoded_preds = [pred.replace("<pad>", "").replace("</s>", "").strip() for pred in decoded_preds]
        decoded_labels = [label.replace("<pad>", "").replace("</s>", "").strip() for label in decoded_labels]
        
        metrics = evaluate_predictions(decoded_labels, decoded_preds)
        
        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "exact_match_count": metrics["exact_match_count"],
            "invalid_generated_count": metrics["invalid_generated_count"]
        }
        
    logger.info("Initializing Seq2SeqTrainer with Production Logging...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ProductionLoggingCallback],
    )
    
    logger.info("Starting production fine-tuning...")
    trainer.train()
    
    # Save best model
    best_model_dir = os.path.join(args.output_dir, "best_model")
    logger.info(f"Saving final best model to {best_model_dir}...")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    # --- Production Level Evaluation on the Unseen Test Dataset ---
    logger.info("Running evaluation on the unseen Test Dataset...")
    test_metrics = trainer.evaluate(eval_dataset=test_tokenized, metric_key_prefix="test")
    logger.info(f"Test Set Evaluation Results: {test_metrics}")
    
    # Save detailed configuration and metrics report
    experiment_config = {
        "base_model": args.model_name,
        "seed": args.seed,
        "epochs": epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "gpu_device": device_name,
    }
    
    metrics_report = {
        "config": experiment_config,
        "final_test_metrics": {
            "test_loss": test_metrics.get("test_loss"),
            "test_precision": test_metrics.get("test_precision"),
            "test_recall": test_metrics.get("test_recall"),
            "test_f1": test_metrics.get("test_f1"),
            "test_exact_match_count": test_metrics.get("test_exact_match_count"),
            "test_invalid_generated_count": test_metrics.get("test_invalid_generated_count"),
        }
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
    logger.info(f"Saved complete metrics and config report to {metrics_path}")
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=4)
        
    logger.info("Production fine-tuning and evaluation complete!")

if __name__ == "__main__":
    main()
