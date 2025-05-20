import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import save_model, load_model

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movies = None

    def fit(self, movies):
        """Fit the TF-IDF model on movie genres."""
        self.movies = movies
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(movies['genres'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        save_model(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        save_model(self.similarity_matrix, 'models/similarity_matrix.pkl')

    def recommend_movies(self, movie_title, top_n=10):
        """Recommend movies similar to the given movie title."""
        if self.tfidf_matrix is None or self.similarity_matrix is None:
            self.tfidf_vectorizer = load_model('models/tfidf_vectorizer.pkl')
            self.similarity_matrix = load_model('models/similarity_matrix.pkl')
        
        # Find movie index
        movie_idx = self.movies[self.movies['title'].str.contains(movie_title, case=False, regex=False, na=False)].index
        if len(movie_idx) == 0:
            return pd.DataFrame(columns=['movieId', 'content_score'])
        movie_idx = movie_idx[0]
        
        # Get similarity scores
        sim_scores = self.similarity_matrix[movie_idx]
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude the movie itself
        
        # Get movie IDs and scores
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        recommendations = pd.DataFrame({
            'movieId': self.movies.iloc[movie_indices]['movieId'],
            'content_score': scores
        })
        return recommendations