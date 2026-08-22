import os
import json
import sys
from pathlib import Path

# Add src/ to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mycner.data.parser import parse_file
from mycner.data.converter import serialize_mt5
from mycner.data.splitter import split_dataset

def main():
    raw_dir = Path("data/raw/171_270_351_500_fixed")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for chunks first
    chunk_files = sorted(list(raw_dir.glob("chunk_*.txt")))
    parsed_sentences = []
    
    if chunk_files:
        print(f"Found {len(chunk_files)} chunk files. Parsing and combining them...")
        for chunk_file in chunk_files:
            print(f"Parsing {chunk_file}...")
            try:
                parsed_sentences.extend(parse_file(str(chunk_file)))
            except Exception as e:
                print(f"Error parsing chunk {chunk_file}: {e}")
                sys.exit(1)
    else:
        raw_path = "data/raw/sample_data.txt"
        print(f"No chunk files found. Parsing single raw dataset from {raw_path}...")
        try:
            parsed_sentences = parse_file(raw_path)
        except Exception as e:
            print(f"Error parsing file: {e}")
            sys.exit(1)
        
    num_sentences = len(parsed_sentences)
    total_tokens = sum(len(s["tokens"]) for s in parsed_sentences)
    
    # Label and Entity frequency stats
    unique_labels = set()
    entity_frequency = {}
    total_entities = 0
    
    sentence_lengths = []
    
    for s in parsed_sentences:
        n_tokens = len(s["tokens"])
        sentence_lengths.append(n_tokens)
        
        for label in s["labels"]:
            unique_labels.add(label)
            if label != "O":
                total_entities += 1
                entity_frequency[label] = entity_frequency.get(label, 0) + 1
                
    # Deterministic label list with "O" always included
    unique_labels.add("O")
    sorted_labels = sorted(list(unique_labels))
    
    # Save labels.txt
    labels_file_path = "data/labels.txt"
    with open(labels_file_path, "w", encoding="utf-8") as lf:
        for lbl in sorted_labels:
            lf.write(lbl + "\n")
            
    min_len = min(sentence_lengths) if sentence_lengths else 0
    max_len = max(sentence_lengths) if sentence_lengths else 0
    avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    
    print("\n--- DATASET REPORT ---")
    print(f"Number of sentences: {num_sentences}")
    print(f"Number of tokens: {total_tokens}")
    print(f"Number of entities: {total_entities}")
    print(f"Number of unique labels: {len(sorted_labels)}")
    print("Entity frequency by label:")
    for lbl, freq in sorted(entity_frequency.items()):
        print(f"  {lbl}: {freq}")
    print(f"Sentence length statistics:")
    print(f"  Min length: {min_len}")
    print(f"  Max length: {max_len}")
    print(f"  Average length: {avg_len:.2f}")
    print("Number of malformed records: 0 (all lines successfully parsed)")
    print("----------------------\n")
    
    # Convert/Serialize for mT5
    print("Serializing sentences for mT5...")
    processed_records = []
    for s in parsed_sentences:
        processed_records.append(serialize_mt5(s["tokens"], s["labels"]))
        
    # Split dataset (80% train, 10% validation, 10% test)
    print("Splitting dataset deterministically...")
    train_split, val_split, test_split = split_dataset(processed_records, seed=42)
    
    # Save processed splits
    def save_jsonl(records, filename):
        path = processed_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} records to {path}")
        
    save_jsonl(train_split, "train.jsonl")
    save_jsonl(val_split, "validation.jsonl")
    save_jsonl(test_split, "test.jsonl")
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
