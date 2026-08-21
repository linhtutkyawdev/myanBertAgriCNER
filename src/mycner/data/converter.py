def serialize_mt5(tokens: list, labels: list) -> dict:
    """
    Converts lists of tokens and labels into mT5 input_text and target_text.
    """
    input_text = " ".join(tokens)
    
    target_parts = []
    for token, label in zip(tokens, labels):
        if label == "O":
            target_parts.append(token)
        else:
            target_parts.append(f"{token}<{label}>")
            
    target_text = " ".join(target_parts)
    
    return {
        "tokens": tokens,
        "labels": labels,
        "input_text": input_text,
        "target_text": target_text
    }
