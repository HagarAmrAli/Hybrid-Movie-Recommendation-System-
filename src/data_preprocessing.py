import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

def load_data(data_path='data'):
    """Load MovieLens dataset files."""
    movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    tags = pd.DataFrame() if not os.path.exists(os.path.join(data_path, 'tags.csv')) else pd.read_csv(os.path.join(data_path, 'tags.csv'))
    links = pd.read_csv(os.path.join(data_path, 'links.csv'))
    return movies, ratings, tags, links

def clean_data(movies, ratings, tags, links):
    """Clean and preprocess the dataset."""
    movies['genres'] = movies['genres'].fillna('Unknown')
    tags['tag'] = tags['tag'].fillna('Unknown') if not tags.empty else tags
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    ratings = ratings.drop_duplicates(['userId', 'movieId'])
    tags = tags.drop_duplicates(['userId', 'movieId', 'tag']) if not tags.empty else tags
    return movies, ratings, tags, links

def preprocess_data(data_path='data'):
    """Load and preprocess all data."""
    movies, ratings, tags, links = load_data(data_path)
    movies, ratings, tags, links = clean_data(movies, ratings, tags, links)
    return movies, ratings, tags, links