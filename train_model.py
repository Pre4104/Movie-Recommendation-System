"""
Movie Recommendation System - Model Training Script
------------------------------------------------------
This script:
1. Loads TMDB 5000 movies + credits datasets
2. Cleans and processes the data
3. Builds a 'tags' column combining overview, genres, keywords, cast, crew
4. Vectorizes tags using CountVectorizer
5. Computes cosine similarity between all movies
6. Saves movies.pkl and similarity.pkl for use in the Streamlit app

Run this ONCE first:  python train_model.py
Make sure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in the same folder.
"""

import ast
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def convert(obj):
    """Extract 'name' field from a JSON-like string (used for genres, keywords)."""
    if not obj or not isinstance(obj, str):
        return []
    return [i['name'] for i in ast.literal_eval(obj)]


def convert_cast(obj):
    """Extract top 3 cast member names from a JSON-like string."""
    if not obj or not isinstance(obj, str):
        return []
    return [i['name'] for i in ast.literal_eval(obj)[:3]]


def fetch_director(obj):
    """Extract the director's name from the crew JSON-like string."""
    if not obj or not isinstance(obj, str):
        return []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []


def collapse(L):
    """Remove spaces within each item so multi-word names become single tokens.
    e.g. 'Sam Worthington' -> 'SamWorthington'
    """
    return [i.replace(" ", "") for i in L]


def main():
    print("Loading datasets...")
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge on title
    movies = movies.merge(credits, on='title')

    # Keep only relevant columns
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    print("Parsing genres and keywords...")
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    print("Extracting top 3 cast members...")
    movies['cast'] = movies['cast'].apply(convert_cast)

    print("Extracting director...")
    movies['crew'] = movies['crew'].apply(fetch_director)

    print("Cleaning text (removing spaces within names)...")
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)

    print("Building tags column...")
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    movies['tags'] = (
        movies['overview']
        + movies['genres']
        + movies['keywords']
        + movies['cast']
        + movies['crew']
    )
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x)).str.lower()

    # Final dataframe used by the app
    movies = movies[['movie_id', 'title', 'tags']]

    print("Vectorizing tags with CountVectorizer...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()

  print("Computing cosine similarity matrix...")
similarity = cosine_similarity(vectors)
similarity = similarity.astype('float32')  # halves memory vs default float64

print("Saving movies.pkl and similarity.pkl...")
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
    print("\nDone! You can now run: streamlit run app.py")
    print(f"Total movies processed: {len(movies)}")


if __name__ == "__main__":
    main()
