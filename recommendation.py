import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
movies = pd.read_csv("movies.csv")

# Create TF-IDF matrix using movie genres
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute cosine similarity between all movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping from movie title to index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Store movie titles
titles = movies["title"]


def recommend(movie_title):
    """
    Returns the top 20 movies similar to the given movie.
    """

    if movie_title not in indices:
        return ["Movie not found."]

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:21]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices].tolist()


# Test
if __name__ == "__main__":
    movie = "Toy Story (1995)"

    print(f"\nRecommendations for: {movie}\n")

    recommendations = recommend(movie)

    for i, movie in enumerate(recommendations, start=1):
        print(f"{i}. {movie}")