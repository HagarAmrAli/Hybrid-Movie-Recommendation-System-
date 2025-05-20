import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from src.utils import load_data

def compute_metrics(predictions, ground_truth):
    """Compute RMSE, MAE, Precision, Recall, and F1-Score."""
    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
    mae = mean_absolute_error(ground_truth, predictions)
    
    predicted_relevant = predictions >= 4
    actual_relevant = ground_truth >= 4
    true_positives = np.sum(predicted_relevant & actual_relevant)
    predicted_positives = np.sum(predicted_relevant)
    actual_positives = np.sum(actual_relevant)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

def evaluate_model(model, ratings, test_size=0.2):
    """Evaluate the model on test data."""
    train, test = train_test_split(ratings, test_size=test_size, random_state=42)
    model.load_data()
    
    model.collab_recommender.fit(train)
    
    predictions = []
    ground_truth = []
    for _, row in test.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        pred = model.collab_recommender.svd_model.predict(user_id, movie_id).est
        predictions.append(pred)
        ground_truth.append(row['rating'])
    
    return compute_metrics(np.array(predictions), np.array(ground_truth))