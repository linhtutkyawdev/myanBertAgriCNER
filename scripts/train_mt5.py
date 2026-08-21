import os
import sys
import json
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mycner.evaluation.metrics import evaluate_predictions

def read_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def main():
    print("Loading data...")
    train_records = read_jsonl("data/processed/train.jsonl")
    val_records = read_jsonl("data/processed/validation.jsonl")
    
    # Read unique labels to add to tokenizer
    with open("data/labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    
    # We add labels like <CROP>, <FARM_OP>, etc. (excluding 'O') as special tokens
    special_tokens = [f"<{lbl}>" for lbl in labels if lbl != "O"]
    print(f"Entities to add as special tokens: {special_tokens}")
    
    model_name = "google/mt5-small"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"Added {num_added} tokens to the tokenizer.")
    
    print(f"Loading model {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Convert lists of dicts to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)
    
    max_input_length = 128
    max_target_length = 128
    
    def preprocess_function(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        
        # Tokenize targets
        labels_tokens = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels_tokens["input_ids"]
        return model_inputs
        
    print("Preprocessing datasets...")
    train_tokenized = train_dataset.map(preprocess_function, batched=True)
    val_tokenized = val_dataset.map(preprocess_function, batched=True)
    
    # Set up training arguments
    output_dir = "experiments/mt5/checkpoints"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability and configure high-performance settings
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("🚀 CUDA GPU detected! Enabling high-performance training configurations (FP16, batch size 16)...")
        train_batch_size = 16
        eval_batch_size = 16
        use_fp16 = True
        epochs = 5
        logging_steps = 50
    else:
        print("⚠️ No GPU detected. Running on CPU (reduced batch size and logging to prevent overhead)...")
        train_batch_size = 4
        eval_batch_size = 4
        use_fp16 = False
        epochs = 1  # 1 epoch on CPU on 7k+ samples still takes a while, but lets keep it short
        logging_steps = 10
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=use_fp16,
        use_cpu=not cuda_available,
        report_to="none"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Compute metrics for validation during training
    def compute_metrics(eval_preds):
        preds, labels_ids = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        import numpy as np
        # Ensure values are within vocab bounds and non-negative
        preds = np.where((preds >= 0) & (preds < len(tokenizer)), preds, tokenizer.pad_token_id)
        labels_ids = np.where((labels_ids >= 0) & (labels_ids < len(tokenizer)), labels_ids, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
        
        # Strip trailing/leading spaces or special padding tokens if any
        decoded_preds = [pred.replace("<pad>", "").replace("</s>", "").strip() for pred in decoded_preds]
        decoded_labels = [label.replace("<pad>", "").replace("</s>", "").strip() for label in decoded_labels]
        
        metrics = evaluate_predictions(decoded_labels, decoded_preds)
        
        # Flatten report or extract key values
        flat_metrics = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "exact_match_count": metrics["exact_match_count"],
            "invalid_generated_count": metrics["invalid_generated_count"]
        }
        return flat_metrics
        
    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("experiments/mt5/best_model")
    tokenizer.save_pretrained("experiments/mt5/best_model")
    
    # Save experiment config
    experiment_config = {
        "base_model": model_name,
        "seed": 42,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "max_sequence_length": max_input_length
    }
    with open("experiments/mt5/config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=4)
        
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
