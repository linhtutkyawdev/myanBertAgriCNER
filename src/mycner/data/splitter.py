import random

def split_dataset(data: list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42) -> tuple:
    """
    Deterministically splits a dataset list into train, validation, and test sets.
    """
    shuffled = list(data)
    # Use a local Random instance to ensure no global state interference
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]
    
    return train_data, val_data, test_data
