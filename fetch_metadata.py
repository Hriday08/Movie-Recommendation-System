import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.themoviedb.org/3"

MOVIES_FILE = "movies.csv"
OUTPUT_FILE = "movie_metadata.csv"

CHECKPOINT_EVERY = 25

# Small delay to be respectful to TMDB's API.
# TMDB currently documents an upper limit around 40 requests/sec.
REQUEST_DELAY = 0.20

MAX_RETRIES = 5

load_dotenv()

TMDB_TOKEN = os.getenv("TMDB_API_KEY")

if not TMDB_TOKEN:
    raise RuntimeError(
        "TMDB_API_KEY not found.\n"
        "Create Backend/.env and add:\n"
        "TMDB_API_KEY=YOUR_API_READ_ACCESS_TOKEN"
    )

HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json",
    "User-Agent": "MovieRecommendationSystem/1.0",
}

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# HELPERS
# ============================================================

def parse_movie_title(title):
    """
    Convert:
        Toy Story (1995)
    into:
        Toy Story
        1995
    """

    match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title)

    if match:
        clean_title = match.group(1).strip()
        year = int(match.group(2))
        return clean_title, year

    return title.strip(), None


def request_json(url, params=None):
    """
    Robust GET request with retries and 429 handling.
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                url,
                params=params,
                timeout=20
            )

            # ------------------------------------------------
            # Success
            # ------------------------------------------------
            if response.status_code == 200:
                return response.json()

            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------
            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    wait_time = float(retry_after)
                else:
                    wait_time = min(2 ** attempt, 30)

                print(
                    f"  Rate limited. Waiting {wait_time:.1f}s..."
                )

                time.sleep(wait_time)
                continue

            # ------------------------------------------------
            # Temporary server errors
            # ------------------------------------------------
            if response.status_code in [500, 502, 503, 504]:

                wait_time = min(2 ** attempt, 30)

                print(
                    f"  Server error {response.status_code}. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            # ------------------------------------------------
            # Unauthorized
            # ------------------------------------------------
            if response.status_code == 401:

                raise RuntimeError(
                    "TMDB returned 401 Unauthorized.\n"
                    "Make sure .env contains your API Read Access Token "
                    "and NOT the API Key."
                )

            # ------------------------------------------------
            # Other error
            # ------------------------------------------------
            print(
                f"  TMDB error {response.status_code}: "
                f"{response.text[:200]}"
            )

            return None

        except requests.RequestException as e:

            wait_time = min(2 ** attempt, 30)

            print(
                f"  Network error: {e}"
            )

            print(
                f"  Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    return None


def search_movie(title, year=None):
    """
    Search TMDB for a movie.

    Attempts to match both title and release year.
    """

    params = {
        "query": title,
        "include_adult": "false",
        "language": "en-US",
    }

    data = request_json(
        f"{BASE_URL}/search/movie",
        params=params
    )

    if not data:
        return None

    results = data.get("results", [])

    if not results:
        return None

    # --------------------------------------------------------
    # If we know the year, try to find matching year.
    # --------------------------------------------------------

    if year:

        for movie in results:

            release_date = movie.get("release_date", "")

            if release_date:

                try:
                    result_year = int(
                        release_date[:4]
                    )

                    if result_year == year:
                        return movie

                except ValueError:
                    pass

    # --------------------------------------------------------
    # Otherwise use first result.
    # --------------------------------------------------------

    return results[0]


def get_movie_details(tmdb_id):
    """
    Fetch complete movie details.

    append_to_response allows us to get:
        details
        credits
        keywords

    in one request.
    """

    params = {
        "language": "en-US",
        "append_to_response": "credits,keywords"
    }

    return request_json(
        f"{BASE_URL}/movie/{tmdb_id}",
        params=params
    )


def extract_director(credits):
    """
    Extract director names.
    """

    if not credits:
        return ""

    crew = credits.get("crew", [])

    directors = []

    for person in crew:

        if person.get("job") == "Director":

            name = person.get("name")

            if name and name not in directors:
                directors.append(name)

    return "|".join(directors)


def extract_cast(credits, limit=10):
    """
    Extract top cast members.
    """

    if not credits:
        return ""

    cast = credits.get("cast", [])

    names = []

    for person in cast[:limit]:

        name = person.get("name")

        if name:
            names.append(name)

    return "|".join(names)


def extract_keywords(keywords):
    """
    Extract movie keywords.
    """

    if not keywords:
        return ""

    keyword_list = keywords.get("keywords", [])

    names = []

    for keyword in keyword_list:

        name = keyword.get("name")

        if name:
            names.append(name)

    return "|".join(names)


def build_metadata(movie_row):
    """
    Process one MovieLens movie.
    """

    movie_id = int(movie_row["movieId"])
    original_title = str(movie_row["title"])
    genres = str(movie_row["genres"])

    clean_title, year = parse_movie_title(
        original_title
    )

    print(
        f"  Searching: {original_title}"
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_result = search_movie(
        clean_title,
        year
    )

    if not search_result:

        print(
            f"  NOT FOUND: {original_title}"
        )

        return {
            "movieId": movie_id,
            "title": original_title,
            "genres": genres,
            "year": year,
            "tmdb_id": None,
            "overview": "",
            "keywords": "",
            "cast": "",
            "director": "",
            "release_date": "",
            "runtime": None,
            "tmdb_rating": None,
            "poster_path": "",
            "backdrop_path": "",
            "metadata_status": "not_found",
        }

    tmdb_id = search_result.get("id")

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    details = get_movie_details(
        tmdb_id
    )

    if not details:

        print(
            f"  DETAILS FAILED: {original_title}"
        )

        return {
            "movieId": movie_id,
            "title": original_title,
            "genres": genres,
            "year": year,
            "tmdb_id": tmdb_id,
            "overview": search_result.get(
                "overview",
                ""
            ),
            "keywords": "",
            "cast": "",
            "director": "",
            "release_date": search_result.get(
                "release_date",
                ""
            ),
            "runtime": None,
            "tmdb_rating": search_result.get(
                "vote_average"
            ),
            "poster_path": search_result.get(
                "poster_path",
                ""
            ),
            "backdrop_path": search_result.get(
                "backdrop_path",
                ""
            ),
            "metadata_status": "details_failed",
        }

    # --------------------------------------------------------
    # CREDITS
    # --------------------------------------------------------

    credits = details.get(
        "credits",
        {}
    )

    director = extract_director(
        credits
    )

    cast = extract_cast(
        credits,
        limit=10
    )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    keywords = extract_keywords(
        details.get("keywords", {})
    )

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    result = {
        "movieId": movie_id,

        "title": original_title,

        "genres": genres,

        "year": year,

        "tmdb_id": tmdb_id,

        "overview": details.get(
            "overview",
            ""
        ),

        "keywords": keywords,

        "cast": cast,

        "director": director,

        "release_date": details.get(
            "release_date",
            ""
        ),

        "runtime": details.get(
            "runtime"
        ),

        "tmdb_rating": details.get(
            "vote_average"
        ),

        "poster_path": details.get(
            "poster_path",
            ""
        ),

        "backdrop_path": details.get(
            "backdrop_path",
            ""
        ),

        "metadata_status": "success",
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TMDB MOVIE METADATA FETCHER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load MovieLens dataset
    # --------------------------------------------------------

    movies = pd.read_csv(
        MOVIES_FILE
    )

    print(
        f"\nTotal movies: {len(movies)}"
    )

    # --------------------------------------------------------
    # Load existing progress
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        existing = pd.read_csv(
            OUTPUT_FILE
        )

        completed_ids = set(
            existing["movieId"]
            .astype(int)
        )

        print(
            f"Already processed: "
            f"{len(completed_ids)}"
        )

        results = existing.to_dict(
            "records"
        )

    else:

        completed_ids = set()

        results = []

        print(
            "No previous checkpoint found."
        )

    # --------------------------------------------------------
    # Process movies
    # --------------------------------------------------------

    processed_since_save = 0

    total = len(movies)

    for index, (_, movie_row) in enumerate(
        movies.iterrows(),
        start=1
    ):

        movie_id = int(
            movie_row["movieId"]
        )

        # ----------------------------------------------------
        # Skip completed
        # ----------------------------------------------------

        if movie_id in completed_ids:

            continue

        print()
        print(
            f"[{index}/{total}]"
        )

        try:

            result = build_metadata(
                movie_row
            )

            results.append(
                result
            )

            completed_ids.add(
                movie_id
            )

            processed_since_save += 1

        except Exception as e:

            print(
                f"  ERROR processing "
                f"{movie_row['title']}: {e}"
            )

            # Save an error row so the movie
            # can be retried later if desired.

            results.append(
                {
                    "movieId": movie_id,
                    "title": movie_row["title"],
                    "genres": movie_row["genres"],
                    "year": parse_movie_title(
                        str(movie_row["title"])
                    )[1],
                    "tmdb_id": None,
                    "overview": "",
                    "keywords": "",
                    "cast": "",
                    "director": "",
                    "release_date": "",
                    "runtime": None,
                    "tmdb_rating": None,
                    "poster_path": "",
                    "backdrop_path": "",
                    "metadata_status": "error",
                }
            )

            completed_ids.add(
                movie_id
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if processed_since_save >= CHECKPOINT_EVERY:

            df = pd.DataFrame(
                results
            )

            df = df.sort_values(
                "movieId"
            )

            df.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print()
            print(
                f"CHECKPOINT SAVED -> "
                f"{OUTPUT_FILE}"
            )

            print(
                f"Progress: "
                f"{len(completed_ids)}/{total}"
            )

            processed_since_save = 0

        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        time.sleep(
            REQUEST_DELAY
        )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df = df.sort_values(
        "movieId"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = (
        df["metadata_status"]
        == "success"
    ).sum()

    not_found = (
        df["metadata_status"]
        == "not_found"
    ).sum()

    failed = (
        df["metadata_status"]
        != "success"
    ).sum()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Not found / failed: {failed}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()