"""Hybrid Movie Recommendation Engine V4.

Uses TMDB-enriched metadata for weighted TF-IDF content similarity,
biased latent-factor collaborative filtering, Bayesian quality scoring,
and user-facing recommendation explanations.

Personalized hybrid weights:
    60% collaborative filtering
    25% content preference
    15% quality
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "movies.csv"
RATINGS_PATH = BASE_DIR / "ratings.csv"
METADATA_PATH = BASE_DIR / "movie_metadata.csv"

movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)

# -------------------------------------------------------------------
# Merge TMDB metadata
# -------------------------------------------------------------------
if METADATA_PATH.exists():
    metadata = pd.read_csv(METADATA_PATH)
    extra = [
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
    extra = [c for c in extra if c in metadata.columns]
    metadata = metadata[extra].drop_duplicates("movieId")
    movies = movies.merge(metadata, on="movieId", how="left")

for col in [
    "overview",
    "keywords",
    "cast",
    "director",
    "release_date",
    "poster_path",
    "backdrop_path",
]:
    if col not in movies:
        movies[col] = ""
    movies[col] = movies[col].fillna("").astype(str)

for col in ["runtime", "tmdb_rating"]:
    if col not in movies:
        movies[col] = np.nan

movies["title"] = movies["title"].fillna("").astype(str)
movies["genres"] = movies["genres"].fillna("").astype(str)
movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
movies["title_clean"] = (
    movies["title"]
    .str.replace(r"\(\d{4}\)", "", regex=True)
    .str.strip()
)

for col in ["keywords", "cast", "director"]:
    movies[col] = movies[col].str.replace("|", " ", regex=False)

# -------------------------------------------------------------------
# Weighted content model
# -------------------------------------------------------------------
CONTENT_FEATURES = {
    "title_clean": 0.15,
    "genres_clean": 0.25,
    "overview": 0.30,
    "keywords": 0.15,
    "cast": 0.10,
    "director": 0.05,
}

blocks = []

for column, weight in CONTENT_FEATURES.items():
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
    )

    # Be robust if a metadata field has no usable vocabulary.
    try:
        matrix = vectorizer.fit_transform(movies[column])
        blocks.append(matrix * np.sqrt(weight))
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            continue
        raise

if not blocks:
    raise RuntimeError("No usable content features were found.")

tfidf_matrix = hstack(blocks, format="csr")

indices = pd.Series(
    movies.index,
    index=movies["title"].str.lower(),
).drop_duplicates()

# -------------------------------------------------------------------
# Quality model
# -------------------------------------------------------------------
movie_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(
        columns={
            "mean": "avg_rating",
            "count": "rating_count",
        }
    )
)

movies = movies.merge(movie_stats, on="movieId", how="left")

GLOBAL_MEAN = float(ratings["rating"].mean())

movies["avg_rating"] = movies["avg_rating"].fillna(GLOBAL_MEAN)
movies["rating_count"] = movies["rating_count"].fillna(0)

MIN_VOTES = 20.0

movies["bayesian_rating"] = (
    (
        movies["rating_count"]
        / (movies["rating_count"] + MIN_VOTES)
    )
    * movies["avg_rating"]
    + (
        MIN_VOTES
        / (movies["rating_count"] + MIN_VOTES)
    )
    * GLOBAL_MEAN
)

movies["quality_score"] = np.clip(
    (movies["bayesian_rating"] - 1.0) / 4.0,
    0.0,
    1.0,
)

# -------------------------------------------------------------------
# Collaborative filtering
# -------------------------------------------------------------------
@dataclass
class MatrixFactorization:
    n_factors: int = 40
    epochs: int = 18
    learning_rate: float = 0.008
    regularization: float = 0.04
    random_state: int = 42

    def fit(self, ratings_df: pd.DataFrame):
        self.user_ids = pd.Index(ratings_df["userId"].unique())
        self.movie_ids = pd.Index(ratings_df["movieId"].unique())

        self.user_to_idx = {
            v: i for i, v in enumerate(self.user_ids)
        }
        self.movie_to_idx = {
            v: i for i, v in enumerate(self.movie_ids)
        }

        rng = np.random.default_rng(self.random_state)

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
                self.user_to_idx[u],
                self.movie_to_idx[m],
                float(r),
            )
            for u, m, r in ratings_df[
                ["userId", "movieId", "rating"]
            ].itertuples(index=False, name=None)
        ]

        for _ in range(self.epochs):
            rng.shuffle(triples)

            for u, i, rating in triples:
                pu = self.user_factors[u]
                qi = self.movie_factors[i]

                pred = (
                    self.global_mean
                    + self.user_bias[u]
                    + self.movie_bias[i]
                    + np.dot(pu, qi)
                )

                error = rating - pred

                # Copy vectors before updating either factor.
                pu_old = pu.copy()
                qi_old = qi.copy()

                self.user_bias[u] += self.learning_rate * (
                    error
                    - self.regularization * self.user_bias[u]
                )

                self.movie_bias[i] += self.learning_rate * (
                    error
                    - self.regularization * self.movie_bias[i]
                )

                self.user_factors[u] += self.learning_rate * (
                    error * qi_old
                    - self.regularization * pu_old
                )

                self.movie_factors[i] += self.learning_rate * (
                    error * pu_old
                    - self.regularization * qi_old
                )

        return self

    def predict_all_for_user(
        self,
        user_id: int,
        movie_ids: np.ndarray,
    ):
        if user_id not in self.user_to_idx:
            return np.full(
                len(movie_ids),
                self.global_mean,
                dtype=np.float32,
            )

        u = self.user_to_idx[user_id]

        result = np.full(
            len(movie_ids),
            self.global_mean + self.user_bias[u],
            dtype=np.float32,
        )

        known = np.array(
            [m in self.movie_to_idx for m in movie_ids]
        )

        if known.any():
            idxs = np.array(
                [
                    self.movie_to_idx[m]
                    for m in movie_ids[known]
                ]
            )

            result[known] += self.movie_bias[idxs]
            result[known] += (
                self.movie_factors[idxs]
                @ self.user_factors[u]
            )

        return np.clip(result, 0.5, 5.0)


collaborative_model = MatrixFactorization().fit(ratings)

user_rated = (
    ratings.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _title_index(movie_title: str):
    key = movie_title.strip().lower()

    if key in indices:
        return int(indices[key])

    matches = movies[
        movies["title_clean"].str.lower() == key
    ]

    if not matches.empty:
        return int(matches.index[0])

    contains = movies[
        movies["title"]
        .str.lower()
        .str.contains(
            re.escape(key),
            regex=True,
            na=False,
        )
    ]

    return (
        int(contains.index[0])
        if not contains.empty
        else None
    )


def _content_scores(movie_idx: int):
    return linear_kernel(
        tfidf_matrix[movie_idx],
        tfidf_matrix,
    ).ravel()


def _normalize(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    lo = values.min()
    hi = values.max()

    if hi - lo < 1e-12:
        return np.zeros_like(values)

    return (values - lo) / (hi - lo)


def _poster_url(path, size="w500"):
    if not path or str(path).lower() == "nan":
        return None

    return f"https://image.tmdb.org/t/p/{size}{path}"


def _movie_year(row):
    match = re.search(
        r"\((\d{4})\)",
        str(row["title"]),
    )
    return int(match.group(1)) if match else None


def _clean_list(text):
    if not text:
        return []

    return [
        item.strip()
        for item in re.split(r"[|,]", str(text))
        if item.strip()
    ]


def _top_overlap(user_values, movie_values, limit=3):
    user_set = {
        str(x).strip().lower()
        for x in user_values
        if str(x).strip()
    }

    movie_items = [
        str(x).strip()
        for x in movie_values
        if str(x).strip()
    ]

    matches = []

    for item in movie_items:
        if item.lower() in user_set:
            matches.append(item)

    return matches[:limit]


def _user_preference_context(user_id: int):
    """Extract interpretable preference signals from highly-rated movies."""
    if user_id not in user_rated:
        return {
            "liked_movies": [],
            "genres": [],
            "directors": [],
            "keywords": [],
            "liked_count": 0,
        }

    user_ratings = ratings[
        ratings["userId"] == user_id
    ][["movieId", "rating"]]

    liked = user_ratings[
        user_ratings["rating"] >= 4.0
    ].sort_values(
        "rating",
        ascending=False,
    )

    liked_movies = []
    genres = []
    directors = []
    keywords = []

    movie_rows = movies.set_index("movieId")

    for movie_id, rating in liked.itertuples(
        index=False,
    ):
        if movie_id not in movie_rows.index:
            continue

        row = movie_rows.loc[movie_id]

        liked_movies.append(
            {
                "title": str(row["title"]),
                "rating": float(rating),
            }
        )

        genres.extend(
            _clean_list(row["genres"])
        )

        directors.extend(
            _clean_list(row["director"])
        )

        keywords.extend(
            _clean_list(row["keywords"])
        )

    return {
        "liked_movies": liked_movies,
        "genres": list(dict.fromkeys(genres)),
        "directors": list(dict.fromkeys(directors)),
        "keywords": list(dict.fromkeys(keywords)),
        "liked_count": len(liked_movies),
    }


def _explanation(
    row,
    content_score,
    predicted_rating,
    quality_score,
    user_id=None,
):
    """Generate a transparent, data-backed explanation."""
    genres = _clean_list(row["genres"])
    movie_directors = _clean_list(row["director"])
    movie_keywords = _clean_list(row["keywords"])

    content_reasons = []

    if genres:
        content_reasons.append(
            f"matches genres such as {', '.join(genres[:4])}"
        )

    if movie_directors:
        content_reasons.append(
            f"features director {', '.join(movie_directors[:2])}"
        )

    if content_score >= 0.80:
        content_strength = "very strong"
    elif content_score >= 0.60:
        content_strength = "strong"
    elif content_score >= 0.40:
        content_strength = "moderate"
    else:
        content_strength = "light"

    preference_match = []
    liked_movies = []

    if user_id is not None:
        context = _user_preference_context(user_id)

        preference_genres = _top_overlap(
            context["genres"],
            genres,
        )
        preference_directors = _top_overlap(
            context["directors"],
            movie_directors,
        )
        preference_keywords = _top_overlap(
            context["keywords"],
            movie_keywords,
        )

        if preference_genres:
            preference_match.append(
                "matches genres you have rated highly: "
                + ", ".join(preference_genres)
            )

        if preference_directors:
            preference_match.append(
                "matches directors you have rated highly: "
                + ", ".join(preference_directors)
            )

        if preference_keywords:
            preference_match.append(
                "matches themes/keywords from movies you rated highly: "
                + ", ".join(preference_keywords)
            )

        liked_movies = [
            item["title"]
            for item in context["liked_movies"][:3]
        ]

    if not preference_match:
        preference_match.append(
            "your collaborative-filtering profile predicts "
            f"{predicted_rating:.2f}/5"
        )

    quality_label = (
        "strong movie-quality signal"
        if quality_score >= 0.70
        else "solid movie-quality signal"
        if quality_score >= 0.55
        else "moderate movie-quality signal"
    )

    bullets = [
        f"{content_strength.capitalize()} content match "
        f"({content_score:.1%})",
        *preference_match[:2],
        f"Predicted rating: {predicted_rating:.2f}/5",
        quality_label,
    ]

    # Short human-readable reason for existing frontend compatibility.
    short_reason = "; ".join(
        content_reasons[:2]
        + preference_match[:1]
    )

    if not short_reason:
        short_reason = (
            f"{content_strength} match with your learned preferences"
        )

    if not short_reason.endswith("."):
        short_reason += "."

    return {
        "summary": short_reason,
        "why_youll_like_it": bullets[:4],
        "content_match": {
            "score": round(float(content_score), 4),
            "strength": content_strength,
        },
        "preference_match": preference_match[:3],
        "predicted_rating": round(
            float(predicted_rating),
            2,
        ),
        "quality": {
            "score": round(
                float(quality_score),
                4,
            ),
            "rating_count": int(row["rating_count"]),
            "average_rating": round(
                float(row["avg_rating"]),
                2,
            ),
        },
        "based_on_liked_movies": liked_movies,
    }


def _movie_payload(
    row,
    content_score,
    predicted_rating,
    hybrid_score,
    user_id=None,
):
    runtime = row.get("runtime")
    runtime = (
        int(runtime)
        if pd.notna(runtime)
        else None
    )

    tmdb_rating = row.get("tmdb_rating")
    tmdb_rating = (
        round(float(tmdb_rating), 2)
        if pd.notna(tmdb_rating)
        else None
    )

    quality_score = float(
        row["quality_score"]
    )

    explanation = _explanation(
        row,
        float(content_score),
        float(predicted_rating),
        quality_score,
        user_id,
    )

    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row["genres"]),
        "year": _movie_year(row),
        "overview": str(row.get("overview", "")),
        "keywords": str(row.get("keywords", "")),
        "cast": str(row.get("cast", "")),
        "director": str(row.get("director", "")),
        "runtime": runtime,
        "avg_rating": round(
            float(row["avg_rating"]),
            2,
        ),
        "rating_count": int(row["rating_count"]),
        "tmdb_rating": tmdb_rating,
        "poster_url": _poster_url(
            row.get("poster_path", "")
        ),
        "backdrop_url": _poster_url(
            row.get("backdrop_path", ""),
            "w1280",
        ),
        "content_score": round(
            float(content_score),
            4,
        ),
        "predicted_rating": round(
            float(predicted_rating),
            2,
        ),
        "hybrid_score": round(
            float(hybrid_score),
            4,
        ),
        "reason": explanation["summary"],
        "explanation": explanation,
    }


# -------------------------------------------------------------------
# Recommendations
# -------------------------------------------------------------------
def recommend(
    movie_title: str,
    user_id: int | None = None,
    limit: int = 20,
):
    idx = _title_index(movie_title)

    if idx is None:
        return []

    content = _content_scores(idx)

    candidate_mask = np.ones(
        len(movies),
        dtype=bool,
    )
    candidate_mask[idx] = False

    if (
        user_id is not None
        and user_id in user_rated
    ):
        candidate_mask &= ~movies[
            "movieId"
        ].isin(
            user_rated[user_id]
        ).to_numpy()

    candidate_idx = np.flatnonzero(
        candidate_mask
    )

    content_values = content[candidate_idx]
    content_norm = _normalize(
        content_values
    )

    quality = movies.iloc[
        candidate_idx
    ]["quality_score"].to_numpy(float)

    if user_id is not None:
        movie_ids = movies.iloc[
            candidate_idx
        ]["movieId"].to_numpy()

        predicted = (
            collaborative_model
            .predict_all_for_user(
                user_id,
                movie_ids,
            )
        )

        collab_norm = np.clip(
            (predicted - 1.0) / 4.0,
            0.0,
            1.0,
        )

        # Seed-based personalized recommendation:
        # content similarity is the primary signal.
        hybrid = (
            0.55 * content_norm
            + 0.35 * collab_norm
            + 0.10 * quality
        )
    else:
        predicted = np.full(
            len(candidate_idx),
            GLOBAL_MEAN,
        )
        hybrid = (
            0.75 * content_norm
            + 0.25 * quality
        )

    order = np.argsort(-hybrid)[:limit]

    return [
        _movie_payload(
            movies.iloc[movie_idx],
            content_values[pos],
            predicted[pos],
            hybrid[pos],
            user_id,
        )
        for pos, movie_idx in zip(
            order,
            candidate_idx[order],
        )
    ]


def recommend_for_user(
    user_id: int,
    limit: int = 20,
):
    """Personalized hybrid recommendations.

    Final validated personalized weights:
        60% collaborative filtering
        25% content preference
        15% quality
    """
    if user_id not in collaborative_model.user_to_idx:
        return []

    rated_ids = user_rated.get(
        user_id,
        set(),
    )

    candidate_mask = ~movies[
        "movieId"
    ].isin(rated_ids).to_numpy()

    candidate_idx = np.flatnonzero(
        candidate_mask
    )

    if len(candidate_idx) == 0:
        return []

    candidate_movie_ids = movies.iloc[
        candidate_idx
    ]["movieId"].to_numpy()

    # ---------------------------------------------------------------
    # 1. Collaborative preference
    # ---------------------------------------------------------------
    predicted = (
        collaborative_model
        .predict_all_for_user(
            user_id,
            candidate_movie_ids,
        )
    )

    collab_norm = np.clip(
        (predicted - 1.0) / 4.0,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------------
    # 2. Content preference profile
    # ---------------------------------------------------------------
    user_ratings = ratings[
        ratings["userId"] == user_id
    ][["movieId", "rating"]]

    liked_indices = []
    weights = []

    movie_lookup = {
        int(movie_id): idx
        for idx, movie_id in enumerate(
            movies["movieId"]
        )
    }

    for movie_id, rating in user_ratings.itertuples(
        index=False
    ):
        if rating < 4.0:
            continue

        idx = movie_lookup.get(
            int(movie_id)
        )

        if idx is None:
            continue

        liked_indices.append(idx)
        weights.append(
            max(float(rating) - 3.0, 0.1)
        )

    if liked_indices:
        liked_matrix = tfidf_matrix[
            np.asarray(
                liked_indices,
                dtype=int,
            )
        ]

        weights_array = np.asarray(
            weights,
            dtype=float,
        )

        weighted_profile = (
            liked_matrix
            .multiply(weights_array[:, None])
            .sum(axis=0)
        )

        weighted_profile = np.asarray(
            weighted_profile
        )

        user_profile = csr_matrix(
            weighted_profile
        )

        user_profile = normalize(
            user_profile,
            norm="l2",
        )

        content_scores = linear_kernel(
            user_profile,
            tfidf_matrix[candidate_idx],
        ).ravel()

        content_norm = _normalize(
            content_scores
        )
    else:
        content_norm = np.zeros(
            len(candidate_idx),
            dtype=float,
        )

    # ---------------------------------------------------------------
    # 3. Quality
    # ---------------------------------------------------------------
    quality = movies.iloc[
        candidate_idx
    ]["quality_score"].to_numpy(
        dtype=float
    )

    # ---------------------------------------------------------------
    # 4. FINAL VALIDATED HYBRID
    # ---------------------------------------------------------------
    hybrid = (
        0.60 * collab_norm
        + 0.25 * content_norm
        + 0.15 * quality
    )

    order = np.argsort(-hybrid)[:limit]

    return [
        _movie_payload(
            movies.iloc[movie_idx],
            content_norm[pos],
            predicted[pos],
            hybrid[pos],
            user_id,
        )
        for pos, movie_idx in zip(
            order,
            candidate_idx[order],
        )
    ]


# -------------------------------------------------------------------
# Movie details / search
# -------------------------------------------------------------------
def get_movie(movie_id: int):
    matches = movies[
        movies["movieId"] == movie_id
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    runtime = row.get("runtime")
    runtime = (
        int(runtime)
        if pd.notna(runtime)
        else None
    )

    tmdb_rating = row.get("tmdb_rating")
    tmdb_rating = (
        round(float(tmdb_rating), 2)
        if pd.notna(tmdb_rating)
        else None
    )

    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row["genres"]),
        "year": _movie_year(row),
        "overview": str(row.get("overview", "")),
        "keywords": str(row.get("keywords", "")),
        "cast": str(row.get("cast", "")),
        "director": str(row.get("director", "")),
        "runtime": runtime,
        "avg_rating": round(
            float(row["avg_rating"]),
            2,
        ),
        "rating_count": int(row["rating_count"]),
        "tmdb_rating": tmdb_rating,
        "poster_url": _poster_url(
            row.get("poster_path", "")
        ),
        "backdrop_url": _poster_url(
            row.get("backdrop_path", ""),
            "w1280",
        ),
    }


def search_movies(
    query: str,
    limit: int = 10,
):
    q = query.strip().lower()

    if not q:
        return []

    mask = movies[
        "title"
    ].str.lower().str.contains(
        re.escape(q),
        regex=True,
        na=False,
    )

    found = (
        movies[mask]
        .sort_values(
            ["rating_count", "avg_rating"],
            ascending=False,
        )
        .head(limit)
    )

    return [
        get_movie(int(row["movieId"]))
        for _, row in found.iterrows()
    ]


if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID MOVIE RECOMMENDER V4 - EXPLAINABLE")
    print("=" * 70)
    print(f"Movies loaded: {len(movies)}")
    print(f"Ratings loaded: {len(ratings)}")
    print(
        f"TMDB metadata: "
        f"{'YES' if METADATA_PATH.exists() else 'NO'}"
    )
    print("\nPersonalized recommendations for user 1:")

    for item in recommend_for_user(1, 10):
        print(
            f"{item['title']} | "
            f"content={item['content_score']} | "
            f"predicted={item['predicted_rating']} | "
            f"hybrid={item['hybrid_score']}"
        )
        print(f"  Why: {item['reason']}")
