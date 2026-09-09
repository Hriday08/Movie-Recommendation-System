🎬 Movie Recommendation System

A personalized hybrid movie recommendation system that combines content-based filtering, collaborative filtering, and quality-aware ranking to recommend movies based on both movie characteristics and user rating behavior.

The project is designed to demonstrate how multiple machine-learning signals can be combined into a practical recommendation engine with an interactive web interface.

✨ What This Project Does

The system answers two related questions:

"If I like this movie, what else might I like?"
"Based on my rating history, what movies are likely to interest me?"

Instead of relying on a single recommendation technique, the system combines multiple signals:

                    Movie Data
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
     Content Model  Collaborative  Quality
       TF-IDF        Filtering      Prior
          │            │            │
          └────────────┼────────────┘
                       ▼
                Hybrid Ranker
                       │
                       ▼
              Personalized Results
🧠 Machine Learning Approach
1. Content-Based Recommendation

Movie information is converted into numerical features using TF-IDF.

The content representation uses:

Title
Genres
Overview
Keywords
Cast
Director

Different fields are given different weights so that important movie information contributes appropriately.

The system then uses cosine similarity to find movies whose content is similar to the selected movie or to the user's learned content preference profile.

2. Collaborative Filtering

The system learns from the user–movie rating matrix.

A biased latent-factor matrix factorization model is trained using historical ratings.

The model learns latent representations of:

Users
Movies

These learned representations are used to predict ratings for movies a user has not rated.

This is the main component that allows the system to learn preference patterns from user behavior rather than only comparing movie descriptions.

3. Quality-Aware Ranking

A Bayesian quality/popularity prior is used as an additional signal.

It considers:

Average rating
Number of ratings

This helps prevent movies with very few ratings from being ranked unrealistically high simply because their average rating is high.

4. Hybrid Recommendation

The final recommendation score combines the different signals.

For personalized recommendations:

60%  Collaborative Filtering
25%  Content Preference
15%  Quality

This allows the system to balance:

What similar users liked + what the user likes + how reliable the movie's rating is

For movie-seeded recommendations, the system also combines content, collaborative, and quality signals with weights tuned separately for that recommendation path.

🔍 Why You'll Like This

The recommendation system is designed to be explainable.

For recommended movies, the API can expose information such as:

Content match
Predicted rating
Hybrid score
User preference match
Highly-rated movies that influenced the preference profile
Matching genres
Matching directors
Matching keywords
Overall rating and rating count

Example:

Why you'll like this

✓ Strong content match
✓ Similar to movies you rated highly
✓ Predicted rating: 4.1 / 5
✓ Strong genre/director/keyword overlap

This makes the recommendation process easier to understand instead of treating the model as a black box.

🎯 Interactive Recommendation Experience

The frontend is built to make the recommendation process visible.

Search

Search for a movie such as:

Toy

The application provides matching movie choices.

Select a Movie

Selecting a movie triggers the recommendation engine and generates a new recommendation set.

Refresh Recommendations

The homepage supports refreshing the recommendation set.

The refresh interaction lets the user explore different recommendations while maintaining relevance to the current recommendation context.

This makes the recommendation system feel interactive rather than like a static list of movies.

Personalized Recommendations

The user can also request recommendations based directly on their rating history.

Movie Details

Selecting a recommendation opens a detailed view containing information such as:

Poster
Rating
Genres
Overview
Director
Cast
Predicted rating
Content similarity
Hybrid score
Recommendation explanation
🏗️ System Architecture
                         ┌─────────────────────┐
                         │      Frontend       │
                         │      React + Vite   │
                         └──────────┬──────────┘
                                    │
                                  HTTP
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         │        REST API     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                     ┌───────────────────────────┐
                     │   Recommendation Engine   │
                     ├───────────────────────────┤
                     │ TF-IDF Content Model      │
                     │ Matrix Factorization     │
                     │ Bayesian Quality Prior    │
                     │ Hybrid Ranker             │
                     │ Explainability            │
                     └─────────────┬─────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
          Movie Metadata                         Ratings
          + TMDB enrichment                   User feedback
📁 Project Structure
Movies_Recommendation_System/
│
├── Backend/
│   ├── main.py
│   ├── recommendation.py
│   ├── movies.csv
│   ├── ratings.csv
│   ├── movie_metadata.csv
│   ├── .env
│   └── requirements.txt
│
├── Frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── ...
│
└── README.md
📊 Dataset

The original movie dataset contains:

10,329 movies
105,339 ratings
668 users

The movie data contains fields such as:

movieId
title
genres

The rating data contains:

userId
movieId
rating
timestamp

Movie metadata was enriched using TMDB information, including fields such as:

Overview
Keywords
Cast
Director
Release date
Runtime
TMDB rating
Poster
Backdrop

The enrichment process successfully obtained metadata for approximately 98% of the movies.

⚙️ Model Configuration
TF-IDF
stop_words       = english
ngram_range      = (1, 2)
min_df           = 2
sublinear_tf     = True
max_features     = 100000

Weighted feature contributions:

Feature	Weight
Title	15%
Genres	25%
Overview	30%
Keywords	15%
Cast	10%
Director	5%
Matrix Factorization
latent factors       = 40
training epochs      = 18
learning rate        = 0.008
regularization       = 0.04
random state         = 42

The collaborative model uses a biased latent-factor matrix-factorization approach trained with SGD.

📈 Evaluation

The recommendation system was evaluated using a temporal train/test split.

The collaborative filtering model achieved approximately:

RMSE = 0.89

The hybrid model was also evaluated using ranking metrics including:

Precision@5
Precision@10
Precision@20
Recall@5
Recall@10
Recall@20

During weight tuning, the personalized hybrid configuration of:

60% Collaborative
25% Content
15% Quality

performed better than the previously tested weighting configuration on the evaluation split.

The best tested configuration achieved approximately:

Precision@10 = 0.0223
Recall@20    = 0.0359

These metrics are intended to measure recommendation quality on the held-out evaluation data rather than represent a percentage accuracy of individual recommendations.

🚀 Running the Backend

Navigate to the backend:

cd Backend

Install dependencies:

pip install -r requirements.txt

Start FastAPI:

uvicorn main:app --reload

The API will run at:

http://127.0.0.1:8000

FastAPI documentation is available at:

http://127.0.0.1:8000/docs
💻 Running the Frontend

Navigate to the frontend:

cd Frontend

Install dependencies:

npm install

Start the development server:

npm run dev

Then open the local Vite URL shown in the terminal, typically:

http://localhost:5173
🔌 API Endpoints
Health Check
GET /
Search Movies
GET /search?query=dark&limit=10
Movie Details
GET /movie/{movie_id}
Personalized Recommendations
GET /recommend/user/{user_id}?limit=20
Movie-Based Recommendations
GET /recommend/{movie_name}?user_id={user_id}&limit=20

The recommendation response can include:

movieId
title
genres
year
overview
keywords
cast
director
runtime
avg_rating
rating_count
tmdb_rating
poster_url
backdrop_url
content_score
predicted_rating
hybrid_score
reason
explanation
🔐 TMDB Configuration

Movie metadata and artwork are enriched using TMDB.

Create a .env file inside Backend/:

TMDB_API_KEY=YOUR_TMDB_READ_ACCESS_TOKEN

Do not commit .env or expose your API credentials publicly.

Add this to .gitignore:

.env
__pycache__/
*.pyc
node_modules/
🧪 Example Recommendation Flow
User searches:
        "Toy"
             │
             ▼
     Search candidates
             │
             ▼
       Select movie
             │
             ▼
   Content similarity
             +
   Collaborative prediction
             +
      Quality signal
             │
             ▼
       Hybrid ranking
             │
             ▼
      Recommended movies
             │
             ▼
      Refresh / explore
🛠️ Tech Stack
Machine Learning
Python
NumPy
Pandas
Scikit-learn
SciPy
Backend
FastAPI
Uvicorn
Python-dotenv
Requests
Frontend
React
Vite
CSS
Data / External API
MovieLens-style movie and rating data
TMDB metadata and artwork
💡 Key Features
🎯 Personalized recommendations
🤖 Collaborative filtering
🎬 Content-based filtering
🧠 Latent-factor matrix factorization
🔀 Hybrid ranking
⭐ Predicted ratings
📊 Bayesian quality-aware ranking
🔍 Movie search
💬 Recommendation explanations
🖼️ TMDB movie posters and metadata
❤️ Favorites
🔄 Interactive recommendation refresh
📱 Responsive movie-focused interface
⚡ FastAPI backend
📈 Offline model evaluation
🔮 Future Improvements

Possible future extensions include:

Real-time user feedback and incremental personalization
More advanced ranking models
Diversity-aware re-ranking
Cold-start onboarding for new users
Neural recommendation models
A/B testing of recommendation strategies
Online evaluation from user interactions
Better handling of unseen/new movies
👨‍💻 Project Goal

The goal of this project is to build a recommendation system that goes beyond simple movie similarity.

It combines learned user/movie patterns, movie-content representations, and quality signals into a single recommendation pipeline, while exposing the reasoning behind the recommendations through an interactive application.

📌 Disclaimer

Movie metadata and artwork are provided through TMDB. This project is intended for educational and portfolio purposes.
