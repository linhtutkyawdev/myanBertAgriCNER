import re
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def parse_mt5_output(text: str) -> list:
    """
    Parses a generated mT5 string back into a list of dictionaries with 'text' and 'label'.
    Example: "စပါး<CROP> ရာတွင် ယူရီးယား<FERT>" ->
    [{'text': 'စပါး', 'label': 'CROP'}, {'text': 'ရာတွင်', 'label': 'O'}, {'text': 'ယူရီးယား', 'label': 'FERT'}]
    """
    entities = []
    parts = text.strip().split()
    for part in parts:
        match = re.match(r"^(.+?)<([A-Z_]+)>$", part)
        if match:
            token = match.group(1)
            label = match.group(2)
            entities.append({"text": token, "label": label})
        else:
            entities.append({"text": part, "label": "O"})
    return entities


def evaluate_predictions(references: list, predictions: list) -> dict:
    """
    Calculates entity-level precision, recall, and F1 metrics using seqeval.
    Each item in references and predictions can be parsed back into lists of labels.
    To use seqeval, we convert sequence of labels to BIO or simply B- prefix for all entities.
    Because seqeval expects BIO/BILOU, we prefix each non-O label with 'B-' (since mT5 doesn't use I- labels,
    treating each labeled word as a single B-LABEL is standard).
    """
    true_sequences = []
    pred_sequences = []
    invalid_count = 0
    exact_match_count = 0
    
    for ref_str, pred_str in zip(references, predictions):
        ref_parsed = parse_mt5_output(ref_str)
        pred_parsed = parse_mt5_output(pred_str)
        
        # Check if they match exactly (excluding minor formatting)
        if ref_str.strip() == pred_str.strip():
            exact_match_count += 1
            
        # Extract true labels
        true_lbls = []
        for item in ref_parsed:
            lbl = item["label"]
            if lbl != "O":
                true_lbls.append(f"B-{lbl}")
            else:
                true_lbls.append("O")
                
        # Extract pred labels
        pred_lbls = []
        # If lengths differ, we align predicted labels to reference tokens where possible,
        # or pad/truncate the predictions list to match reference length.
        if len(pred_parsed) != len(ref_parsed):
            invalid_count += 1
            # Fallback alignment: match by index up to minimum length, then pad/truncate
            for i in range(len(ref_parsed)):
                if i < len(pred_parsed):
                    lbl = pred_parsed[i]["label"]
                    if lbl != "O":
                        pred_lbls.append(f"B-{lbl}")
                    else:
                        pred_lbls.append("O")
                else:
                    pred_lbls.append("O")
        else:
            for item in pred_parsed:
                lbl = item["label"]
                if lbl != "O":
                    pred_lbls.append(f"B-{lbl}")
                else:
                    pred_lbls.append("O")
                    
        true_sequences.append(true_lbls)
        pred_sequences.append(pred_lbls)
        
    p = precision_score(true_sequences, pred_sequences)
    r = recall_score(true_sequences, pred_sequences)
    f1 = f1_score(true_sequences, pred_sequences)
    
    # Per-label metrics
    report = classification_report(true_sequences, pred_sequences, output_dict=True)
    
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "exact_match_count": exact_match_count,
        "invalid_generated_count": invalid_count,
        "report": report
    }
