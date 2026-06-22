# 🎬 Movie Recommendation System

A content-based movie recommendation engine that suggests similar movies based on plot, genre, cast, and crew — built with Python, scikit-learn, and Streamlit.

**[Live Demo →](#)** 

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

> `movies.pkl` and `similarity.pkl` are tracked using **Git LFS** (Large File Storage) since they exceed typical version control size — see [Why Git LFS?](#why-git-lfs) below.

---

## Setup & Usage

### 1. Install Git LFS (one-time, if you don't have it)
```bash
git lfs install
```

### 2. Clone the repo
```bash
git clone https://github.com/<your-username>/movie-recommender.git
cd movie-recommender
```
Git LFS will automatically pull `movies.pkl` and `similarity.pkl` during clone.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
Opens automatically at `http://localhost:8501`

---

### Re-training the model (optional)

If you want to rebuild the model yourself instead of using the committed `.pkl` files:

1. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project folder
3. Run:
```bash
python train_model.py
```
This regenerates `movies.pkl` and `similarity.pkl` locally.

---

## Why Git LFS?

The trained similarity matrix is a large binary file (~90MB as float32). Committing it directly to git would bloat the repository's history with a non-diffable binary blob. **Git LFS** solves this by storing large files outside the normal git object database while keeping them version-controlled and instantly available on clone — including for Streamlit Cloud's deployment pipeline, which pulls LFS files automatically during build.

This keeps the repo:
- Lightweight to clone for casual browsing
- Reproducible (you can still regenerate the files via `train_model.py`)
- Deployment-ready without recomputing the model on every cold start

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

## Author

Built by **B S Lakshmi Prerana** as part of an ML/AI portfolio project, exploring content-based recommendation systems and NLP-based feature engineering.
