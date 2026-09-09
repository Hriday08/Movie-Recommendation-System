import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(BASE_DIR, "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "ratings.csv")
METADATA_PATH = os.path.join(BASE_DIR, "movie_metadata.csv")

N_FACTORS, EPOCHS = 40, 18
LEARNING_RATE, REGULARIZATION = 0.008, 0.04
THRESHOLD, MIN_USER_RATINGS = 4.0, 5
TOP_K = (5, 10, 20)

WEIGHTS = [
    ("45_45_10", .45, .45, .10),
    ("50_40_10", .50, .40, .10),
    ("55_35_10_CURRENT", .55, .35, .10),
    ("60_30_10", .60, .30, .10),
    ("65_25_10", .65, .25, .10),
    ("50_35_15", .50, .35, .15),
    ("55_30_15", .55, .30, .15),
    ("60_25_15", .60, .25, .15),
    ("45_40_15", .45, .40, .15),
]


@dataclass
class MatrixFactorization:
    n_factors: int = N_FACTORS
    epochs: int = EPOCHS
    lr: float = LEARNING_RATE
    reg: float = REGULARIZATION
    random_state: int = 42

    def fit(self, df):
        rng = np.random.default_rng(self.random_state)
        users, items = df.userId.unique(), df.movieId.unique()
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {m: i for i, m in enumerate(items)}
        self.global_mean = float(df.rating.mean())
        self.ub = np.zeros(len(users)); self.ib = np.zeros(len(items))
        self.uf = rng.normal(0, .1, (len(users), self.n_factors))
        self.if_ = rng.normal(0, .1, (len(items), self.n_factors))
        triples = [(self.user_to_idx[u], self.item_to_idx[m], float(r))
                   for u, m, r in df[["userId","movieId","rating"]].itertuples(index=False)]
        for e in range(self.epochs):
            rng.shuffle(triples)
            for u, i, r in triples:
                pred = self.global_mean + self.ub[u] + self.ib[i] + np.dot(self.uf[u], self.if_[i])
                err = r - pred
                uv, iv = self.uf[u].copy(), self.if_[i].copy()
                self.ub[u] += self.lr * (err - self.reg * self.ub[u])
                self.ib[i] += self.lr * (err - self.reg * self.ib[i])
                self.uf[u] += self.lr * (err * iv - self.reg * uv)
                self.if_[i] += self.lr * (err * uv - self.reg * iv)
            print(f"Epoch {e+1:02d}/{self.epochs}")
        return self

    def predict_all_for_user(self, user_id, movie_ids):
        if user_id not in self.user_to_idx:
            return np.full(len(movie_ids), self.global_mean)
        u = self.user_to_idx[user_id]
        out = []
        for m in movie_ids:
            if m in self.item_to_idx:
                i = self.item_to_idx[m]
                p = self.global_mean + self.ub[u] + self.ib[i] + np.dot(self.uf[u], self.if_[i])
            else:
                p = self.global_mean + self.ub[u]
            out.append(np.clip(p, .5, 5.0))
        return np.asarray(out)


def clean(df):
    for c in ["title_clean","genres","overview","keywords","cast","director"]:
        if c not in df: df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df


def split_temporal(ratings):
    ratings = ratings.sort_values(["userId","timestamp"])
    train, test = [], []
    for _, g in ratings.groupby("userId", sort=False):
        if len(g) < MIN_USER_RATINGS:
            train.append(g); continue
        cut = min(max(1, int(np.floor(len(g)*.8))), len(g)-1)
        train.append(g.iloc[:cut]); test.append(g.iloc[cut:])
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def build_content(movies):
    """Build a robust weighted TF-IDF matrix.

    Some TMDB metadata fields can be completely empty or contain only
    stop-words. Those blocks are skipped instead of crashing.
    """
    blocks = []
    feature_config = [
        ("title_clean", .15),
        ("genres", .25),
        ("overview", .30),
        ("keywords", .15),
        ("cast", .10),
        ("director", .05),
    ]

    for col, weight in feature_config:
        text = movies[col].fillna("").astype(str).str.strip()

        # Skip a feature if there is no usable text at all.
        if text.eq("").all():
            print(f"  Skipping {col}: column is completely empty")
            continue

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )

        try:
            block = vectorizer.fit_transform(text)
        except ValueError as exc:
            if "empty vocabulary" in str(exc):
                print(f"  Skipping {col}: no usable vocabulary")
                continue
            raise

        print(f"  {col}: {block.shape[1]:,} features")
        blocks.append(block * weight)

    if not blocks:
        raise RuntimeError(
            "No usable text features were found in movie_metadata.csv. "
            "Check that title/genres/overview/keywords/cast/director contain data."
        )

    return normalize(hstack(blocks).tocsr(), norm="l2")


def norm_scores(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.ptp(x) < 1e-12: return np.zeros_like(x)
    return (x - x.min()) / np.ptp(x)


def quality_scores(train, movies, min_votes=20):
    s = train.groupby("movieId").rating.agg(["mean","count"])
    gm = float(train.rating.mean())
    out = []
    for m in movies.movieId:
        if m in s.index:
            mean, count = float(s.loc[m,"mean"]), float(s.loc[m,"count"])
            q = count/(count+min_votes)*mean + min_votes/(count+min_votes)*gm
        else: q = gm
        out.append(q)
    return np.clip((np.asarray(out)-1)/4, 0, 1)


def profiles(train, movies, matrix):
    lookup = {int(m): i for i,m in enumerate(movies.movieId)}
    result = {}
    for uid, g in train.groupby("userId"):
        liked = g[g.rating >= THRESHOLD]
        idx, w = [], []
        for m,r in liked[["movieId","rating"]].itertuples(index=False):
            if int(m) in lookup:
                idx.append(lookup[int(m)]); w.append(max(float(r)-3, .1))
        if idx:
            p = matrix[np.asarray(idx)].multiply(np.asarray(w)[:,None]).sum(axis=0)
            result[int(uid)] = normalize(csr_matrix(np.asarray(p)), norm="l2")
    return result


def evaluate(train, test, movies, matrix, quality, model, user_profiles, weights):
    name, cw, tw, qw = weights
    mids = movies.movieId.to_numpy()
    lookup = {int(m): i for i,m in enumerate(mids)}
    seen = train.groupby("userId").movieId.apply(set).to_dict()
    relevant = test[test.rating >= THRESHOLD].groupby("userId").movieId.apply(set).to_dict()
    ps = {k:0.0 for k in TOP_K}; rs = {k:0.0 for k in TOP_K}; n=0

    for uid, rel_ids in relevant.items():
        if uid not in model.user_to_idx: continue
        cand_mask = np.ones(len(movies), dtype=bool)
        for m in seen.get(uid,set()):
            if int(m) in lookup: cand_mask[lookup[int(m)]] = False
        ci = np.flatnonzero(cand_mask)
        if len(ci)==0: continue
        cm = mids[ci]
        cf = np.clip((model.predict_all_for_user(uid,cm)-1)/4,0,1)
        profile = user_profiles.get(int(uid))
        content = np.zeros(len(ci)) if profile is None else norm_scores(
            linear_kernel(profile, matrix[ci]).ravel()
        )
        hybrid = cw*cf + tw*content + qw*quality[ci]
        ranked = cm[np.argsort(-hybrid)]
        rel = {int(m) for m in rel_ids if int(m) in lookup and int(m) not in seen.get(uid,set())}
        if not rel: continue
        n += 1
        for k in TOP_K:
            hits = len(set(map(int,ranked[:k])) & rel)
            ps[k] += hits/k; rs[k] += hits/len(rel)

    if not n: return None
    r = {"config":name,"cf_weight":cw,"content_weight":tw,"quality_weight":qw,"users_evaluated":n}
    for k in TOP_K:
        r[f"precision_at_{k}"] = ps[k]/n
        r[f"recall_at_{k}"] = rs[k]/n
    r["selection_score"] = .5*r["precision_at_10"] + .5*r["recall_at_20"]
    return r


def main():
    print("="*70); print("HYBRID MOVIE RECOMMENDER - WEIGHT TUNING"); print("="*70)
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    if "rating" not in ratings and "ratings" in ratings: ratings.rename(columns={"ratings":"rating"}, inplace=True)

    # Preserve title and genres from the original MovieLens dataset.
    # Only merge TMDB-derived metadata so empty TMDB fields cannot overwrite them.
    movies["title_clean"] = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )
    movies["genres"] = movies["genres"].fillna("").astype(str)

    if os.path.exists(METADATA_PATH):
        meta = pd.read_csv(METADATA_PATH)
        tmdb_cols = ["movieId", "overview", "keywords", "cast", "director"]
        available = [c for c in tmdb_cols if c in meta.columns]
        movies = movies.merge(
            meta[available].drop_duplicates("movieId"),
            on="movieId",
            how="left",
        )
        movies = clean(movies)
        print(f"TMDB metadata loaded: {len(meta):,} movies")
        print(
            "Feature coverage: "
            f"title_clean={movies['title_clean'].ne('').sum():,}, "
            f"genres={movies['genres'].ne('').sum():,}, "
            f"overview={movies['overview'].ne('').sum():,}, "
            f"keywords={movies['keywords'].ne('').sum():,}, "
            f"cast={movies['cast'].ne('').sum():,}, "
            f"director={movies['director'].ne('').sum():,}"
        )
    else:
        movies = clean(movies)
        print("WARNING: movie_metadata.csv not found.")

    train,test = split_temporal(ratings)
    print(f"Training ratings: {len(train):,}"); print(f"Testing ratings:  {len(test):,}")

    print("\nTraining collaborative model...")
    model = MatrixFactorization().fit(train)

    print("\nBuilding content model...")
    matrix = build_content(movies)
    print(f"Content matrix shape: {matrix.shape}")

    print("\nBuilding profiles and quality scores...")
    user_profiles = profiles(train,movies,matrix)
    quality = quality_scores(train,movies)

    print("\nTesting weight combinations...")
    results=[]
    for i,w in enumerate(WEIGHTS,1):
        print(f"[{i}/{len(WEIGHTS)}] {w[0]}: CF={w[1]:.0%}, Content={w[2]:.0%}, Quality={w[3]:.0%}")
        r=evaluate(train,test,movies,matrix,quality,model,user_profiles,w)
        if r:
            results.append(r)
            print(f"  P@10={r['precision_at_10']:.4f} | R@20={r['recall_at_20']:.4f} | Score={r['selection_score']:.4f}")

    df=pd.DataFrame(results).sort_values("selection_score",ascending=False)
    out=os.path.join(BASE_DIR,"weight_tuning_results.csv"); df.to_csv(out,index=False)
    print("\n"+"="*70); print("RESULTS (BEST FIRST)"); print("="*70)
    print(df.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    best=df.iloc[0]
    print("\nBEST:")
    print(f"CF={best.cf_weight:.0%}, Content={best.content_weight:.0%}, Quality={best.quality_weight:.0%}")
    print(f"P@10={best.precision_at_10:.4f}, R@20={best.recall_at_20:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
