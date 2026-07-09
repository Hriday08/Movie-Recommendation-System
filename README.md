# 🎬 AI-Based Movie Recommendation System

An AI-powered full-stack Movie Recommendation System that recommends similar movies using **Content-Based Filtering**. The application leverages **TF-IDF Vectorization** and **Cosine Similarity** on the **MovieLens Dataset** to generate intelligent movie recommendations through a modern React interface and FastAPI backend.

---

## ✨ Features

- 🎥 Search movies by title
- 🤖 AI-powered movie recommendations
- ⚡ FastAPI REST API
- 🌐 Modern React + Vite frontend
- 📊 Content-Based Recommendation Engine
- 📚 MovieLens Dataset Integration
- 🔍 Real-time recommendation results

---

## 🖼️ Demo

> **Live Demo:** *(Add Vercel Link Here)*

> **API Documentation:** *(Add Render/FastAPI Link Here)*

---

## 📸 Screenshots

### Home Page

<img src="screenshots/home.png" width="900"/>

### Recommendation Results

<img src="screenshots/results.png" width="900"/>

---

## 🧠 How It Works

1. User enters a movie title.
2. React sends a request to the FastAPI backend.
3. The backend searches the MovieLens dataset.
4. Movie genres are transformed using **TF-IDF Vectorization**.
5. **Cosine Similarity** calculates similarity between movies.
6. The top similar movies are returned.
7. Recommendations are displayed instantly on the frontend.

---

## 🛠️ Tech Stack

### Frontend
- React
- Vite
- JavaScript
- HTML5
- CSS3

### Backend
- Python
- FastAPI
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Content-Based Filtering

### Dataset
- MovieLens Dataset

---

## 📂 Project Structure

```text
Movie-Recommendation-System
│
├── Backend
│   ├── main.py
│   ├── recommendation.py
│   ├── movies.csv
│   ├── ratings.csv
│   └── requirements.txt
│
├── Frontend
│   ├── src
│   ├── public
│   ├── package.json
│   └── vite.config.js
│
├── screenshots
│   ├── Backend.png
│   └── Frontend1&2.png
│
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
```

Move into the project directory.

```bash
cd movie-recommendation-system
```

---

### Backend Setup

```bash
cd Backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend:

```
http://127.0.0.1:8000
```

Swagger API:

```
http://127.0.0.1:8000/docs
```

---

### Frontend Setup

```bash
cd Frontend
npm install
npm run dev
```

Frontend:

```
http://localhost:5173
```

---

## 📊 Recommendation Technique

This project implements a **Content-Based Recommendation System**.

The recommendation engine compares movies based on their **genres** using:

- TF-IDF Vectorization
- Cosine Similarity

Movies with the highest similarity scores are recommended to the user.

---

## 🚀 Future Enhancements

- 🔐 User Authentication
- ⭐ User Ratings
- ❤️ Favorite Movies
- 🎭 Movie Posters (TMDb API)
- 🤝 Collaborative Filtering (SVD)
- 🔀 Hybrid Recommendation System
- 📈 Trending Movies Dashboard
- 📱 Responsive Mobile UI

---

## 👨‍💻 Author

**Hriday Siddharth**

Computer Science Undergraduate

📧 hridaysiddharth08@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/hriday-siddharth/


---

## ⭐ Support

If you found this project useful, please consider **starring ⭐ the repository**.
