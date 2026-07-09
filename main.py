from fastapi import FastAPI
from recommendation import recommend
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommendation import recommend

# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="AI-powered Movie Recommendation System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home route
@app.get("/")
def home():
    return {
        "message": "Welcome to the Movie Recommendation API!"
    }

# Recommendation route
@app.get("/recommend/{movie_name}")
def get_recommendations(movie_name: str):
    recommendations = recommend(movie_name)

    return {
        "movie": movie_name,
        "recommendations": recommendations
    }