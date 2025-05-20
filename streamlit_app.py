import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.hybrid_model import HybridRecommender
from src.utils import load_data

# Page configuration
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

# Load data
@st.cache_data
def load_cached_data():
    movies, ratings, _, _ = load_data()
    return movies, ratings

movies, ratings = load_cached_data()

# Initialize recommender
recommender = HybridRecommender()
recommender.load_data()

# Sidebar for inputs
st.sidebar.header("Customize Your Recommendations")

# User ID selection
user_ids = sorted(ratings['userId'].unique())
user_id = st.sidebar.selectbox("Select User ID:", user_ids, index=0)

# Movie title search with autocomplete
movie_titles = movies['title'].tolist()
movie_title = st.sidebar.text_input("Search for a Movie (optional):", "")
if movie_title:
    suggestions = [title for title in movie_titles if movie_title.lower() in title.lower()]
    if suggestions:
        st.sidebar.write("Suggestions:", suggestions[:5])
    else:
        st.sidebar.warning("No movies found matching your search.")

# Genre filter
genres = sorted(set(g for genres in movies['genres'].str.split(' ') for g in genres if g))
selected_genres = st.sidebar.multiselect("Filter by Genres:", genres)

# Weight sliders
st.sidebar.subheader("Adjust Model Weights")
content_weight = st.sidebar.slider("Content-Based Weight:", 0.0, 1.0, 0.4, 0.1)
collab_weight = st.sidebar.slider("Collaborative Filtering Weight:", 0.0, 1.0, 0.6, 0.1)
if abs(content_weight + collab_weight - 1.0) > 0.01:
    st.sidebar.warning("Weights should sum to 1.0. Current sum: {:.2f}".format(content_weight + collab_weight))

# Number of recommendations
top_n = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)

# Main content
st.header("Your Movie Recommendations")

if st.sidebar.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        # Update recommender weights
        recommender.content_weight = content_weight
        recommender.collab_weight = collab_weight

        # Get recommendations with additional scores
        recommendations = recommender.get_hybrid_recommendations(
            user_id=user_id,
            movie_title=movie_title if movie_title else None,
            top_n=top_n * 2  # Get extra to filter by genres
        )

        # Filter by selected genres
        if selected_genres:
            recommendations = recommendations[
                recommendations['genres'].apply(
                    lambda x: any(g in x.split(' ') for g in selected_genres)
                )
            ].head(top_n)

        if recommendations.empty:
            st.error("No recommendations found. Try adjusting your inputs.")
        else:
            st.write("### Recommended Movies")
            st.dataframe(
                recommendations[['title', 'genres', 'hybrid_score']].style.format({
                    'hybrid_score': '{:.3f}'
                }),
                use_container_width=True
            )

            # Visualization 1: Recommendation Scores Bar Chart
            st.subheader("Recommendation Scores")
            fig_scores = px.bar(
                recommendations,
                x='title',
                y='hybrid_score',
                title="Hybrid Scores of Recommended Movies",
                labels={'title': 'Movie Title', 'hybrid_score': 'Hybrid Score'},
                color='hybrid_score',
                color_continuous_scale='Viridis'
            )
            fig_scores.update_layout(xaxis_tickangle=45, showlegend=False)
            st.plotly_chart(fig_scores, use_container_width=True)

            # Visualization 2: Genre Distribution Pie Chart
            st.subheader("Genre Distribution in Recommendations")
            genre_counts = pd.Series([
                genre for genres in recommendations['genres'].str.split(' ') for genre in genres
            ]).value_counts()
            fig_genres = px.pie(
                names=genre_counts.index,
                values=genre_counts.values,
                title="Genre Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_genres, use_container_width=True)

            # Visualization 3 (New): Movie Popularity vs. Recommendation Score
            st.subheader("Popularity vs. Recommendation Score")
            popularity = ratings.groupby('movieId').size().reset_index(name='popularity')
            rec_with_pop = recommendations.merge(popularity, on='movieId', how='left').fillna(0)
            rec_with_pop['dominant_genre'] = rec_with_pop['genres'].apply(lambda x: x.split(' ')[0])
            fig_popularity = px.scatter(
                rec_with_pop,
                x='popularity',
                y='hybrid_score',
                size='hybrid_score',
                color='dominant_genre',
                hover_data=['title'],
                title="Popularity vs. Recommendation Score",
                labels={'popularity': 'Number of Ratings', 'hybrid_score': 'Hybrid Score'}
            )
            st.plotly_chart(fig_popularity, use_container_width=True)

            # Visualization 4 (New): Content vs. Collaborative Contribution
            st.subheader("Content vs. Collaborative Contribution")
            # Recompute raw scores for visualization
            content_scores = recommender.content_recommender.recommend_movies(
                movie_title if movie_title else recommendations['title'].iloc[0], top_n=top_n * 2
            )
            collab_scores = recommender.collab_recommender.predict_user_ratings(user_id)
            hybrid_scores = content_scores.merge(collab_scores, on='movieId', how='left')
            hybrid_scores = hybrid_scores[hybrid_scores['movieId'].isin(recommendations['movieId'])]
            hybrid_scores['content_contribution'] = content_weight * hybrid_scores['content_score'].fillna(0)
            hybrid_scores['collab_contribution'] = collab_weight * hybrid_scores['predicted_rating'].fillna(0)
            hybrid_scores = hybrid_scores.merge(movies[['movieId', 'title']], on='movieId')
            fig_contributions = go.Figure(data=[
                go.Bar(
                    name='Content-Based',
                    x=hybrid_scores['title'],
                    y=hybrid_scores['content_contribution'],
                    marker_color='#FF9999'
                ),
                go.Bar(
                    name='Collaborative Filtering',
                    x=hybrid_scores['title'],
                    y=hybrid_scores['collab_contribution'],
                    marker_color='#66B2FF'
                )
            ])
            fig_contributions.update_layout(
                barmode='stack',
                title="Content vs. Collaborative Contribution to Hybrid Score",
                xaxis_title="Movie Title",
                yaxis_title="Score Contribution",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig_contributions, use_container_width=True)

            # Optional: Download recommendations
            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="Download Recommendations",
                data=csv,
                file_name="movie_recommendations.csv",
                mime="text/csv"
            )

# Visualization 5: User Ratings Histogram
st.subheader("User Ratings Distribution")
user_ratings = ratings[ratings['userId'] == user_id][['rating']] if user_id else ratings[['rating']]
fig_ratings = px.histogram(
    user_ratings,
    x='rating',
    nbins=10,
    title=f"Rating Distribution for {'User ' + str(user_id) if user_id else 'All Users'}",
    labels={'rating': 'Rating', 'count': 'Frequency'},
    color_discrete_sequence=['#636EFA']
)
st.plotly_chart(fig_ratings, use_container_width=True)

# Visualization 6 (New): Top Genres by User Preference
st.subheader("Top Genres by User Preference")
high_rated = ratings[(ratings['userId'] == user_id) & (ratings['rating'] >= 4)]
if not high_rated.empty:
    high_rated_movies = high_rated.merge(movies[['movieId', 'genres']], on='movieId')
    user_genre_counts = pd.Series([
        genre for genres in high_rated_movies['genres'].str.split(' ') for genre in genres
    ]).value_counts().head(5)
    fig_user_genres = px.bar(
        x=user_genre_counts.index,
        y=user_genre_counts.values,
        title=f"Top Genres for User {user_id} (Ratings ≥ 4)",
        labels={'x': 'Genre', 'y': 'Count'},
        color=user_genre_counts.index,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
else:
    genre_counts = pd.Series([
        genre for genres in movies['genres'].str.split(' ') for genre in genres
    ]).value_counts().head(5)
    fig_user_genres = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Top Genres in Dataset (No High Ratings for User)",
        labels={'x': 'Genre', 'y': 'Count'},
        color=genre_counts.index,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
st.plotly_chart(fig_user_genres, use_container_width=True)

# Visualization 7 (New): User Activity Over Time
st.subheader("User Activity Over Time")
user_ratings_time = ratings[ratings['userId'] == user_id][['timestamp']]
if not user_ratings_time.empty and 'timestamp' in user_ratings_time.columns:
    user_ratings_time['date'] = pd.to_datetime(user_ratings_time['timestamp'], unit='s')
    user_ratings_time['year_month'] = user_ratings_time['date'].dt.to_period('M').astype(str)
    activity_counts = user_ratings_time['year_month'].value_counts().sort_index()
    fig_activity = px.line(
        x=activity_counts.index,
        y=activity_counts.values,
        title=f"Rating Activity for User {user_id}",
        labels={'x': 'Year-Month', 'y': 'Number of Ratings'},
        markers=True,
        color_discrete_sequence=['#FFCC00']
    )
    fig_activity.update_layout(xaxis_tickangle=45)
else:
    fig_activity = go.Figure()
    fig_activity.add_annotation(
        text="No timestamp data available for User {}".format(user_id),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig_activity.update_layout(title="Rating Activity for User {}".format(user_id))
st.plotly_chart(fig_activity, use_container_width=True)

# Evaluation metrics (optional)
if st.checkbox("Show Model Evaluation Metrics"):
    st.header("Model Performance")
    from src.evaluation import evaluate_model
    with st.spinner("Evaluating model..."):
        metrics = evaluate_model(recommender, ratings)
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        st.table(metrics_df.style.format({'Value': '{:.4f}'}))

        # Visualization 8: Evaluation Metrics Bar Chart
        st.subheader("Evaluation Metrics")
        fig_metrics = go.Figure(data=[
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Value'],
                marker_color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB3E6']
            )
        ])
        fig_metrics.update_layout(
            title="Model Evaluation Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            showlegend=False
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and MovieLens dataset.")