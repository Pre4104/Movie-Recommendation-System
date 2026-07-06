"""
Movie Recommendation System - Streamlit App
------------------------------------------------------
Run this AFTER train_model.py has generated movies.pkl and similarity.pkl

Command to run:
    streamlit run app.py
"""

import pickle

import streamlit as st

# Load the pre-trained model files
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))


def recommend(movie):
    """Return a list of the top 5 most similar movie titles."""
    matches = movies[movies['title'] == movie]

    if matches.empty:
        return []

    movie_index = matches.index[0]
    distances = similarity[movie_index]
    movies_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]


# ----------------- UI -----------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get 5 similar recommendations based on content (genres, cast, crew, plot).")

selected_movie = st.selectbox("Select a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if not recommendations:
        st.error(f"'{selected_movie}' not found in the dataset.")
    else:
        st.subheader("Top 5 Recommendations:")
        for title in recommendations:
            st.write(f"🎥 {title}")