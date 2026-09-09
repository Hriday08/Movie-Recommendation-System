"""
Model Evaluation for Hybrid Movie Recommendation System

Evaluates:
1. Collaborative Filtering -> RMSE
2. Content-Based Recommendation -> Precision@K / Recall@K
3. Collaborative Recommendation -> Precision@K / Recall@K
4. Hybrid Recommendation -> Precision@K / Recall@K

Important:
We split each user's ratings into train/test BEFORE training the
collaborative model. This prevents evaluation leakage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MOVIES_PATH = BASE_DIR / "movies.csv"
RATINGS_PATH = BASE_DIR / "ratings.csv"
METADATA_PATH = BASE_DIR / "movie_metadata.csv"

TEST_RATIO = 0.20
MIN_RATINGS_PER_USER = 5

K_VALUES = [5, 10, 20]

# Ratings >= this value are considered relevant/liked.
RELEVANT_RATING = 4.0

# To keep evaluation reasonably fast on a laptop.
# Set to None to evaluate every eligible user.
MAX_USERS = None

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID MOVIE RECOMMENDER - MODEL EVALUATION")
print("=" * 70)

movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)

print(f"\nMovies:  {len(movies):,}")
print(f"Ratings: {len(ratings):,}")
print(f"Users:   {ratings['userId'].nunique():,}")


# ============================================================
# LOAD TMDB METADATA
# ============================================================

if METADATA_PATH.exists():
    metadata = pd.read_csv(METADATA_PATH)

    extra_columns = [
        "movieId",
        "overview",
        "keywords",
        "cast",
        "director",
        "release_date",
        "runtime",
        "tmdb_rating",
        "poster_path",
        "backdrop_path",
    ]

    extra_columns = [
        c for c in extra_columns
        if c in metadata.columns
    ]

    metadata = metadata[extra_columns].drop_duplicates("movieId")

    movies = movies.merge(
        metadata,
        on="movieId",
        how="left",
    )

    print(f"TMDB metadata loaded: {len(metadata):,} movies")

else:
    print("WARNING: movie_metadata.csv not found.")
    print("Continuing with original movie metadata.")


# ============================================================
# CLEAN MOVIE DATA
# ============================================================

for column in [
    "overview",
    "keywords",
    "cast",
    "director",
]:
    if column not in movies.columns:
        movies[column] = ""

    movies[column] = (
        movies[column]
        .fillna("")
        .astype(str)
        .str.replace("|", " ", regex=False)
    )

movies["title"] = movies["title"].fillna("").astype(str)
movies["genres"] = movies["genres"].fillna("").astype(str)

movies["genres_clean"] = (
    movies["genres"]
    .str.replace("|", " ", regex=False)
)

movies["title_clean"] = (
    movies["title"]
    .str.replace(r"\(\d{4}\)", "", regex=True)
    .str.strip()
)


# ============================================================
# 1. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 - TRAIN / TEST SPLIT")
print("=" * 70)

# Sort chronologically if timestamp exists.
if "timestamp" in ratings.columns:
    ratings = ratings.sort_values(
        ["userId", "timestamp"]
    )

train_parts = []
test_parts = []

rng = np.random.default_rng(RANDOM_STATE)

for user_id, user_df in ratings.groupby("userId"):

    if len(user_df) < MIN_RATINGS_PER_USER:
        continue

    # Temporal holdout:
    # latest 20% ratings become test data.
    test_size = max(1, int(len(user_df) * TEST_RATIO))

    train_user = user_df.iloc[:-test_size]
    test_user = user_df.iloc[-test_size:]

    train_parts.append(train_user)
    test_parts.append(test_user)


train_ratings = pd.concat(
    train_parts,
    ignore_index=True,
)

test_ratings = pd.concat(
    test_parts,
    ignore_index=True,
)

print(f"Training ratings: {len(train_ratings):,}")
print(f"Testing ratings:  {len(test_ratings):,}")
print(
    f"Train ratio: "
    f"{len(train_ratings) / len(ratings):.2%}"
)
print(
    f"Test ratio:  "
    f"{len(test_ratings) / len(ratings):.2%}"
)


# ============================================================
# 2. COLLABORATIVE FILTERING MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - TRAIN COLLABORATIVE FILTERING MODEL")
print("=" * 70)


@dataclass
class MatrixFactorization:

    n_factors: int = 40
    epochs: int = 18
    learning_rate: float = 0.008
    regularization: float = 0.04
    random_state: int = 42

    def fit(self, ratings_df):

        self.user_ids = pd.Index(
            ratings_df["userId"].unique()
        )

        self.movie_ids = pd.Index(
            ratings_df["movieId"].unique()
        )

        self.user_to_idx = {
            user_id: index
            for index, user_id
            in enumerate(self.user_ids)
        }

        self.movie_to_idx = {
            movie_id: index
            for index, movie_id
            in enumerate(self.movie_ids)
        }

        rng = np.random.default_rng(
            self.random_state
        )

        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)

        self.user_factors = rng.normal(
            0,
            0.08,
            (n_users, self.n_factors),
        )

        self.movie_factors = rng.normal(
            0,
            0.08,
            (n_movies, self.n_factors),
        )

        self.user_bias = np.zeros(
            n_users,
            dtype=np.float32,
        )

        self.movie_bias = np.zeros(
            n_movies,
            dtype=np.float32,
        )

        self.global_mean = float(
            ratings_df["rating"].mean()
        )

        triples = [
            (
                self.user_to_idx[user_id],
                self.movie_to_idx[movie_id],
                float(rating),
            )
            for user_id, movie_id, rating
            in ratings_df[
                ["userId", "movieId", "rating"]
            ].itertuples(
                index=False,
                name=None,
            )
        ]

        for epoch in range(self.epochs):

            rng.shuffle(triples)

            for u, i, rating in triples:

                pu = self.user_factors[u]
                qi = self.movie_factors[i]

                prediction = (
                    self.global_mean
                    + self.user_bias[u]
                    + self.movie_bias[i]
                    + np.dot(pu, qi)
                )

                error = rating - prediction

                self.user_bias[u] += (
                    self.learning_rate
                    * (
                        error
                        - self.regularization
                        * self.user_bias[u]
                    )
                )

                self.movie_bias[i] += (
                    self.learning_rate
                    * (
                        error
                        - self.regularization
                        * self.movie_bias[i]
                    )
                )

                self.user_factors[u] += (
                    self.learning_rate
                    * (
                        error * qi
                        - self.regularization * pu
                    )
                )

                self.movie_factors[i] += (
                    self.learning_rate
                    * (
                        error * pu
                        - self.regularization * qi
                    )
                )

            print(
                f"Epoch {epoch + 1:02d}/{self.epochs}"
            )

        return self

    def predict(self, user_id, movie_id):

        if user_id not in self.user_to_idx:
            return self.global_mean

        if movie_id not in self.movie_to_idx:
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.movie_to_idx[movie_id]

        prediction = (
            self.global_mean
            + self.user_bias[u]
            + self.movie_bias[i]
            + np.dot(
                self.user_factors[u],
                self.movie_factors[i],
            )
        )

        return float(
            np.clip(prediction, 0.5, 5.0)
        )

    def predict_all_for_user(
        self,
        user_id,
        movie_ids,
    ):

        if user_id not in self.user_to_idx:

            return np.full(
                len(movie_ids),
                self.global_mean,
                dtype=np.float32,
            )

        u = self.user_to_idx[user_id]

        predictions = np.full(
            len(movie_ids),
            self.global_mean
            + self.user_bias[u],
            dtype=np.float32,
        )

        known = np.array([
            movie_id in self.movie_to_idx
            for movie_id in movie_ids
        ])

        if known.any():

            indices = np.array([
                self.movie_to_idx[movie_id]
                for movie_id in movie_ids[known]
            ])

            predictions[known] += (
                self.movie_bias[indices]
            )

            predictions[known] += (
                self.movie_factors[indices]
                @ self.user_factors[u]
            )

        return np.clip(
            predictions,
            0.5,
            5.0,
        )


print("\nTraining matrix factorization...")

cf_model = MatrixFactorization().fit(
    train_ratings
)


# ============================================================
# 3. RMSE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - RMSE")
print("=" * 70)

actual = []
predicted = []

for row in test_ratings.itertuples(
    index=False
):

    # Only evaluate users/movies known to training.
    if (
        row.userId not in cf_model.user_to_idx
        or row.movieId not in cf_model.movie_to_idx
    ):
        continue

    actual.append(float(row.rating))

    predicted.append(
        cf_model.predict(
            row.userId,
            row.movieId,
        )
    )

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted,
    )
)

print(f"\nEvaluated ratings: {len(actual):,}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# 4. CONTENT MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - BUILD CONTENT MODEL")
print("=" * 70)

CONTENT_FEATURES = {
    "title_clean": 0.15,
    "genres_clean": 0.25,
    "overview": 0.30,
    "keywords": 0.15,
    "cast": 0.10,
    "director": 0.05,
}

content_blocks = []

for column, weight in CONTENT_FEATURES.items():

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    )

    matrix = vectorizer.fit_transform(
        movies[column]
    )

    content_blocks.append(
        matrix * np.sqrt(weight)
    )

tfidf_matrix = hstack(
    content_blocks,
    format="csr",
)

print(
    f"Content matrix shape: "
    f"{tfidf_matrix.shape}"
)


# ============================================================
# 5. QUALITY SCORE
# ============================================================

train_stats = (
    train_ratings
    .groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .reset_index()
)

train_stats = train_stats.rename(
    columns={
        "mean": "avg_rating",
        "count": "rating_count",
    }
)

movies_eval = movies.copy()

movies_eval = movies_eval.merge(
    train_stats,
    on="movieId",
    how="left",
)

global_mean = float(
    train_ratings["rating"].mean()
)

movies_eval["avg_rating"] = (
    movies_eval["avg_rating"]
    .fillna(global_mean)
)

movies_eval["rating_count"] = (
    movies_eval["rating_count"]
    .fillna(0)
)

MIN_VOTES = 20.0

movies_eval["bayesian_rating"] = (
    (
        movies_eval["rating_count"]
        / (
            movies_eval["rating_count"]
            + MIN_VOTES
        )
    )
    * movies_eval["avg_rating"]
    +
    (
        MIN_VOTES
        / (
            movies_eval["rating_count"]
            + MIN_VOTES
        )
    )
    * global_mean
)

movies_eval["quality_score"] = np.clip(
    (
        movies_eval["bayesian_rating"]
        - 1.0
    ) / 4.0,
    0.0,
    1.0,
)


# ============================================================
# 6. USER RATING HISTORY
# ============================================================

train_user_rated = (
    train_ratings
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

test_user_ratings = (
    test_ratings
    .groupby("userId")
)


# ============================================================
# 7. CONTENT PROFILE
# ============================================================

def build_user_profile(user_id):

    if user_id not in train_user_rated:
        return None

    user_history = train_ratings[
        train_ratings["userId"] == user_id
    ]

    liked = user_history[
        user_history["rating"]
        >= RELEVANT_RATING
    ]

    if liked.empty:
        return None

    liked_indices = []
    weights = []

    for movie_id, rating in liked[
        ["movieId", "rating"]
    ].itertuples(
        index=False,
        name=None,
    ):

        matches = movies_eval.index[
            movies_eval["movieId"]
            == movie_id
        ]

        if len(matches) == 0:
            continue

        liked_indices.append(
            int(matches[0])
        )

        # A 5-star movie gets more influence
        # than a 4-star movie.
        weights.append(
            max(float(rating) - 3.0, 0.1)
        )

    if not liked_indices:
        return None

    liked_matrix = tfidf_matrix[
        np.asarray(
            liked_indices,
            dtype=int,
        )
    ]

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weighted_profile = (
        liked_matrix
        .multiply(weights[:, None])
        .sum(axis=0)
    )

    weighted_profile = np.asarray(
        weighted_profile
    )

    profile = csr_matrix(
        weighted_profile
    )

    return normalize(
        profile,
        norm="l2",
    )


# ============================================================
# 8. RECOMMENDATIONS FOR ONE USER
# ============================================================

def get_rankings(user_id):

    if user_id not in train_user_rated:
        return None

    rated_movies = train_user_rated[
        user_id
    ]

    candidate_mask = (
        ~movies_eval["movieId"]
        .isin(rated_movies)
        .to_numpy()
    )

    candidate_indices = np.flatnonzero(
        candidate_mask
    )

    candidate_movie_ids = (
        movies_eval
        .iloc[candidate_indices]["movieId"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Collaborative score
    # --------------------------------------------------------

    predictions = (
        cf_model.predict_all_for_user(
            user_id,
            candidate_movie_ids,
        )
    )

    collab_norm = np.clip(
        (predictions - 1.0) / 4.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Content score
    # --------------------------------------------------------

    profile = build_user_profile(
        user_id
    )

    if profile is not None:

        content_scores = linear_kernel(
            profile,
            tfidf_matrix[
                candidate_indices
            ],
        ).ravel()

        # Clip because tiny floating-point
        # differences can produce very small negatives.
        content_scores = np.clip(
            content_scores,
            0.0,
            1.0,
        )

    else:

        content_scores = np.zeros(
            len(candidate_indices)
        )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality = (
        movies_eval
        .iloc[candidate_indices]
        ["quality_score"]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_scores = (
        0.55 * collab_norm
        + 0.35 * content_scores
        + 0.10 * quality
    )

    return {
        "movie_ids": candidate_movie_ids,
        "collab": collab_norm,
        "content": content_scores,
        "hybrid": hybrid_scores,
    }


# ============================================================
# 9. METRICS
# ============================================================

def precision_at_k(
    recommended,
    relevant,
    k,
):

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0.0

    hits = len(
        set(recommended)
        & set(relevant)
    )

    return hits / len(recommended)


def recall_at_k(
    recommended,
    relevant,
    k,
):

    if len(relevant) == 0:
        return 0.0

    recommended = recommended[:k]

    hits = len(
        set(recommended)
        & set(relevant)
    )

    return hits / len(relevant)


# ============================================================
# 10. EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - RANKING EVALUATION")
print("=" * 70)

eligible_users = []

for user_id, user_test in test_user_ratings:

    # Only evaluate users with at least one relevant
    # held-out rating.
    relevant = user_test[
        user_test["rating"]
        >= RELEVANT_RATING
    ]

    if len(relevant) == 0:
        continue

    if user_id not in train_user_rated:
        continue

    eligible_users.append(user_id)


if MAX_USERS is not None:

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(eligible_users) > MAX_USERS:

        eligible_users = list(
            rng.choice(
                eligible_users,
                size=MAX_USERS,
                replace=False,
            )
        )


print(
    f"\nUsers evaluated: "
    f"{len(eligible_users):,}"
)

results = {
    "content": {
        k: {
            "precision": [],
            "recall": [],
        }
        for k in K_VALUES
    },
    "collab": {
        k: {
            "precision": [],
            "recall": [],
        }
        for k in K_VALUES
    },
    "hybrid": {
        k: {
            "precision": [],
            "recall": [],
        }
        for k in K_VALUES
    },
}


for count, user_id in enumerate(
    eligible_users,
    start=1,
):

    rankings = get_rankings(
        user_id
    )

    if rankings is None:
        continue

    user_test = test_ratings[
        test_ratings["userId"]
        == user_id
    ]

    relevant_movies = set(
        user_test[
            user_test["rating"]
            >= RELEVANT_RATING
        ]["movieId"]
    )

    movie_ids = rankings["movie_ids"]

    # Sort candidates according to each model.
    content_order = np.argsort(
        -rankings["content"]
    )

    collab_order = np.argsort(
        -rankings["collab"]
    )

    hybrid_order = np.argsort(
        -rankings["hybrid"]
    )

    content_ranked = (
        movie_ids[content_order]
    )

    collab_ranked = (
        movie_ids[collab_order]
    )

    hybrid_ranked = (
        movie_ids[hybrid_order]
    )

    for k in K_VALUES:

        for model_name, ranked in [
            ("content", content_ranked),
            ("collab", collab_ranked),
            ("hybrid", hybrid_ranked),
        ]:

            p = precision_at_k(
                ranked,
                relevant_movies,
                k,
            )

            r = recall_at_k(
                ranked,
                relevant_movies,
                k,
            )

            results[
                model_name
            ][k]["precision"].append(p)

            results[
                model_name
            ][k]["recall"].append(r)

    if count % 25 == 0:
        print(
            f"Processed {count:,}/"
            f"{len(eligible_users):,} users"
        )


# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION RESULTS")
print("=" * 70)

print("\nCollaborative Filtering")
print("-" * 40)
print(f"RMSE: {rmse:.4f}")

for model_name in [
    "content",
    "collab",
    "hybrid",
]:

    print(
        f"\n{model_name.upper()} MODEL"
    )
    print("-" * 40)

    for k in K_VALUES:

        precision = np.mean(
            results[
                model_name
            ][k]["precision"]
        )

        recall = np.mean(
            results[
                model_name
            ][k]["recall"]
        )

        print(
            f"Precision@{k:2d}: "
            f"{precision:.4f}"
        )

        print(
            f"Recall@{k:2d}:    "
            f"{recall:.4f}"
        )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

summary_rows = []

for model_name in [
    "content",
    "collab",
    "hybrid",
]:

    for k in K_VALUES:

        precision = np.mean(
            results[
                model_name
            ][k]["precision"]
        )

        recall = np.mean(
            results[
                model_name
            ][k]["recall"]
        )

        summary_rows.append({
            "model": model_name,
            "k": k,
            "precision": precision,
            "recall": recall,
        })


results_df = pd.DataFrame(
    summary_rows
)

output_path = (
    BASE_DIR
    / "evaluation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print("\n" + "=" * 70)
print(
    f"Results saved to:\n{output_path}"
)
print("=" * 70)