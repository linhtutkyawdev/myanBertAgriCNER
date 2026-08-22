import argparse
import os
import json
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mycner.evaluation.metrics import parse_mt5_output

def predict_mt5(text: str):
    model_path = "experiments/mt5/best_model"
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist. Have you trained the model?")
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Align model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Prepend the task prefix "cner: " to match the fine-tuning input format
    formatted_text = f"cner: {text}"
    
    # Tokenize input and place on device
    inputs = tokenizer(formatted_text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Clean padding and end-of-sequence tokens
    generated_text = generated_text.replace("<pad>", "").replace("</s>", "").strip()
    
    entities = parse_mt5_output(generated_text)
    return {
        "text": text,
        "labeled_text": generated_text,
        "entities": [ent for ent in entities if ent["label"] != "O"]
    }

def main():
    import os
    parser = argparse.ArgumentParser(description="Predict CNER entities.")
    parser.add_argument("--model", type=str, required=True, choices=["mt5"], help="Model type to use.")
    parser.add_argument("--text", type=str, required=True, help="Input Burmese agricultural sentence.")
    
    args = parser.parse_args()
    
    if args.model == "mt5":
        result = predict_mt5(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=4))
    else:
        print(f"Error: Model {args.model} is not supported or not implemented yet.")
        sys.exit(1)

if __name__ == "__main__":
    main()
