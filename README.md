🎬 Movie Recommendation System

A hybrid ML-based movie recommendation system that combines content-based filtering, collaborative filtering, and quality-aware ranking to deliver personalized recommendations.

🧠 ML Pipeline
Movie Data + User Ratings
          ↓
 ┌────────┼─────────┐
 ↓        ↓         ↓
TF-IDF  Matrix     Quality
Content Factorization Prior
 ↓        ↓         ↓
 └────────┼─────────┘
          ↓
    Hybrid Ranking
          ↓
 Personalized Movies

🚀 Features

🤖 Collaborative Filtering using latent-factor matrix factorization
🎬 Content-Based Filtering using TF-IDF + cosine similarity
🔀 Hybrid recommendation ranking
⭐ Predicted user ratings
💬 Explainable recommendations — "Why you'll like this"
🔍 Interactive movie search
🔄 Refreshable recommendations
❤️ Favorites
🖼️ TMDB movie metadata & posters
📊 Model evaluation using RMSE, Precision@K and Recall@K


⚙️ Tech Stack

ML: Python, Pandas, NumPy, Scikit-learn, SciPy
Backend: FastAPI, Uvicorn
Frontend: React, Vite, CSS
Data: Movie ratings + TMDB metadata

📊 Model

Personalized hybrid ranking:

60% Collaborative Filtering
25% Content Similarity
15% Quality

Collaborative filtering achieves approximately 0.89 RMSE on the evaluation setup.

▶️ Run Locally

Backend

cd Backend
pip install -r requirements.txt
uvicorn main:app --reload

Frontend

cd Frontend
npm install
npm run dev

Then open http://localhost:5173.

Built to demonstrate a practical end-to-end ML recommendation system — from data and model training to personalized recommendations and an interactive web application.
