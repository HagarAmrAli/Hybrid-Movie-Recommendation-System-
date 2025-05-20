import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.utils import load_data

class HybridRecommender:
    def __init__(self, content_weight=0.5, collab_weight=0.5):
        self.content_recommender = ContentBasedRecommender()
        self.collab_recommender = CollaborativeFilteringRecommender()
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.movies = None
        self.ratings = None

    def load_data(self):
        """Load and preprocess data."""
        self.movies, self.ratings, _, _ = load_data()
        self.content_recommender.fit(self.movies)
        self.collab_recommender.fit(self.ratings)

    def get_hybrid_recommendations(self, user_id, movie_title=None, top_n=10):
        """Generate hybrid recommendations for a user."""
        if movie_title:
            content_recs = self.content_recommender.recommend_movies(movie_title, top_n * 2)
        else:
            content_recs = pd.DataFrame({
                'movieId': self.movies['movieId'],
                'content_score': [0.5] * len(self.movies)  # Neutral score if no movie
            })

        collab_recs = self.collab_recommender.predict_user_ratings(user_id)
        
        hybrid_recs = content_recs.merge(collab_recs, on='movieId', how='inner')
        hybrid_recs['hybrid_score'] = (
            self.content_weight * hybrid_recs['content_score'] +
            self.collab_weight * hybrid_recs['predicted_rating']
        )
        
        hybrid_recs = hybrid_recs.sort_values('hybrid_score', ascending=False).head(top_n * 2)
        hybrid_recs = hybrid_recs.merge(self.movies[['movieId', 'title', 'genres']], on='movieId')
        
        return hybrid_recs[['movieId', 'title', 'genres', 'content_score', 'predicted_rating', 'hybrid_score']]