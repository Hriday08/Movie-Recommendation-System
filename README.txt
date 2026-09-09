MOVIE RECOMMENDATION SYSTEM - BACKEND V2

Replace the existing Backend/main.py and Backend/recommendation.py with these files.
Copy requirements.txt into Backend/.

Run from the Backend folder:
    pip install -r requirements.txt
    uvicorn main:app --reload

API examples:
    GET /search?query=toy&limit=10
    GET /movie/1
    GET /recommend/Toy%20Story%20%281995%29?limit=20
    GET /recommend/Toy%20Story%20%281995%29?user_id=1&limit=20
    GET /recommend/user/1?limit=20

New ML pipeline:
    TF-IDF(title + genres)
          +
    latent-factor collaborative filtering (SGD matrix factorization)
          +
    Bayesian quality prior
          -> hybrid ranking

The current MovieLens files do not contain plot/overview/poster/cast/director
metadata. Those will be added in the next stage rather than fabricated.
