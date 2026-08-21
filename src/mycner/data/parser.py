import re

def parse_line(line: str, line_no: int) -> dict:
    """
    Parses a single line of TOKEN@LABEL|TOKEN@LABEL|... format into tokens and labels list.
    """
    # Strip whitespace, but do not ignore if it contains characters
    stripped = line.strip()
    if not stripped:
        return {"tokens": [], "labels": []}
    
    # Split by '|'
    parts = stripped.split('|')
    
    # If there's a trailing pipe, the last split element will be empty.
    if len(parts) > 1 and parts[-1] == "":
        parts = parts[:-1]
        
    tokens = []
    labels = []
    
    for part_idx, part in enumerate(parts):
        if not part:
            raise ValueError(f"Malformed entry on line {line_no}: Empty token-label segment found.")
            
        # Segment should have exactly one '@' splitting token and label.
        # But wait! What if token contains '@'? Typically, it splits on the LAST '@' or we find exact parts.
        # Standard format is TOKEN@LABEL. Let's split from the right side once.
        if '@' not in part:
            raise ValueError(f"Malformed entry on line {line_no}: Segment '{part}' lacks '@' separator.")
            
        token, label = part.rsplit('@', 1)
        if not token or not label:
            raise ValueError(f"Malformed entry on line {line_no}: Segment '{part}' has empty token or label.")
            
        tokens.append(token)
        labels.append(label)
        
    return {"tokens": tokens, "labels": labels}


def parse_file(file_path: str) -> list:
    """
    Parses a full dataset file, returning a list of dicts with keys 'tokens' and 'labels'.
    """
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.rstrip("\r\n")
            if not line_str.strip():
                # Ignore truly empty lines
                continue
            try:
                parsed = parse_line(line_str, idx)
                if parsed["tokens"]:
                    sentences.append(parsed)
            except ValueError as e:
                raise ValueError(f"Error parsing {file_path} at line {idx}: {str(e)}")
    return sentences
