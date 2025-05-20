import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pickle

def load_data(data_path='data'):
    """Load data using preprocessing module."""
    from src.data_preprocessing import preprocess_data
    return preprocess_data(data_path)

def save_model(model, path):
    """Save model to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """Load model from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)