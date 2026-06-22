# 🎬 Movie Recommendation System

A content-based movie recommendation engine that suggests similar movies based on plot, genre, cast, and crew — built with Python, scikit-learn, and Streamlit.

**[Live Demo →](#)** *(add your Streamlit Cloud link here after deployment)*

---

## Overview

Pick any movie, and the app instantly recommends 5 similar titles — no user ratings or login required. It works purely off **content**: what a movie is *about*, not how others rated it.

This makes it a **content-based recommender**, as opposed to collaborative filtering systems (like Netflix's "users who watched X also watched Y").

---

## How It Works

1. **Data** — TMDB 5000 Movies + Credits dataset (~4,800 movies)
2. **Feature Engineering** — combines plot overview, genres, keywords, top 3 cast members, and director into a single `tags` field per movie
3. **Vectorization** — converts text tags into numerical vectors using `CountVectorizer` (Bag of Words, 5000 features)
4. **Similarity** — computes pairwise **Cosine Similarity** between every movie vector
5. **Recommendation** — for a selected movie, returns the 5 movies with the highest similarity score

> Cosine similarity was chosen over alternatives like Euclidean distance because it measures the *direction* (pattern of tags) rather than magnitude — so a movie with fewer tags isn't unfairly penalized against one with many.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | pandas |
| Text vectorization | scikit-learn (`CountVectorizer`) |
| Similarity computation | scikit-learn (`cosine_similarity`) |
| Web app | Streamlit |
| Model storage | pickle |

---

## Project Structure

```
movie-recommender/
├── train_model.py        # Data processing + model training (run once)
├── app.py                 # Streamlit web app
├── requirements.txt       # Python dependencies
├── tmdb_5000_movies.csv   # Dataset (not included — see Setup)
└── tmdb_5000_credits.csv  # Dataset (not included — see Setup)
```

> `movies.pkl` and `similarity.pkl` are generated locally by `train_model.py` and are **not committed to this repo** (see [Why no .pkl files?](#why-no-pkl-files)).

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle and place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project folder.

### 4. Train the model (run once)
```bash
python train_model.py
```
This generates `movies.pkl` and `similarity.pkl`.

### 5. Run the app
```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`

---

## Why No `.pkl` Files?

The trained similarity matrix is a large binary file (~90MB+) that:
- Provides no value in version control (can't be diffed or reviewed)
- Risks exceeding GitHub's file size limits as the dataset grows
- Is fully reproducible by running `train_model.py` — there's no reason to track generated artifacts

This follows standard ML engineering practice: **track code and data sources, not generated model files.**

---

## Example

**Input:** `Captain America: Civil War`

**Output:**
```
Captain America: The First Avenger
Captain America: The Winter Soldier
Avengers: Age of Ultron
Iron Man 3
Iron Man 2
```

---

## Possible Improvements

- [ ] Add movie posters via TMDB API for a richer UI
- [ ] Try TF-IDF vectorization as an alternative to CountVectorizer
- [ ] Add a hybrid approach combining content-based + collaborative filtering
- [ ] Deploy with a larger, more recent dataset to cover newer releases

---

## Author

Built by **B S Lakshmi Prerana** as part of an ML/AI portfolio project, exploring content-based recommendation systems and NLP-based feature engineering.
