# RecoAI Movie Recommendation UI

A React/Vite frontend for the existing FastAPI movie recommendation backend.

## Run

1. Start backend:
   `uvicorn main:app --reload`
2. Install frontend dependencies:
   `npm install`
3. Start frontend:
   `npm run dev`

Open the Vite URL shown in the terminal (normally http://localhost:5173).

The UI calls:
- `/recommend/user/{user_id}`
- `/recommend/{movie_name}?user_id={user_id}`
