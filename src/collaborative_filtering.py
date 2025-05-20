import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from surprise import SVD, Dataset, Reader
from src.utils import save_model, load_model

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)
        self.trainset = None
        self.movies = None

    def fit(self, ratings):
        """Train the SVD model on ratings data."""
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        self.trainset = data.build_full_trainset()
        self.svd_model.fit(self.trainset)
        save_model(self.svd_model, 'models/svd_model.pkl')

    def predict_user_ratings(self, user_id):
        """Predict ratings for all movies for a given user."""
        if self.svd_model is None:
            self.svd_model = load_model('models/svd_model.pkl')
        
        # Get all movie IDs
        movie_ids = self.trainset.all_items()
        movie_ids = [self.trainset.to_raw_iid(iid) for iid in movie_ids]
        
        # Predict ratings
        predictions = [self.svd_model.predict(user_id, mid).est for mid in movie_ids]
        return pd.DataFrame({
            'movieId': movie_ids,
            'predicted_rating': predictions
        })
        