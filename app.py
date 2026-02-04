from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn

class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv('movies.csv')
        self.ratings = pd.read_csv('ratings.csv')
        self.tags = pd.read_csv('tags.csv')
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        
    def train(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies['genres'].fillna('')
        )
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_top_movies(self, n: int = 20):
        top_rated = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        top_rated.columns = ['movieId', 'avg_rating', 'rating_count']
        top_rated = top_rated[top_rated['rating_count'] >= 5].sort_values(
            'avg_rating', ascending=False
        ).head(n)
        
        result = self.movies.merge(top_rated, on='movieId', how='inner')
        return result
    
    def get_movies_by_genre(self, genre: str, n: int = 20):
        mask = self.movies['genres'].str.contains(genre, case=False, na=False)
        movies = self.movies[mask].head(n)
        
        movie_ratings = self.ratings.groupby('movieId')['rating'].mean()
        movies = movies.copy()
        movies['avg_rating'] = movies['movieId'].map(movie_ratings)
        return movies.sort_values('avg_rating', ascending=False).head(n)
    
    def get_user_ratings(self, user_id: int, n: int = 50):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        user_movies = user_ratings.merge(
            self.movies, on='movieId', how='left'
        ).sort_values('rating', ascending=False).head(n)
        return user_movies
    
    def get_recommendations_by_ratings(self, user_id: int, n: int = 10):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if user_ratings.empty:
            return self.get_top_movies(n)
        
        rated_movie_ids = user_ratings['movieId'].values
        rated_indices = self.movies[
            self.movies['movieId'].isin(rated_movie_ids)
        ].index.values
        
        if len(rated_indices) == 0:
            return self.get_top_movies(n)
        
        sim_scores = self.similarity_matrix[rated_indices[0]].copy()
        for idx in rated_indices[1:]:
            sim_scores += self.similarity_matrix[idx]
        
        sim_scores = sim_scores / len(rated_indices)
        
        similar_movies_indices = sim_scores.argsort()[::-1]
        
        recommendations = self.movies.iloc[similar_movies_indices].head(n)
        movie_ratings = self.ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
        
        result = recommendations.merge(
            movie_ratings, left_on='movieId', right_index=True, how='left'
        ).rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})
        
        return result
    
    def recommend_by_multiple_genres(self, genres: List[str], n: int = 15):
        mask = pd.Series([False] * len(self.movies))
        for genre in genres:
            mask |= self.movies['genres'].str.contains(genre, case=False, na=False)
        
        result = self.movies[mask].head(n)
        movie_ratings = self.ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
        
        result = result.merge(
            movie_ratings, left_on='movieId', right_index=True, how='left'
        ).rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})
        
        return result.sort_values('avg_rating', ascending=False).head(n)
    
    def get_all_genres(self):
        all_genres = set()
        for genres_str in self.movies['genres']:
            all_genres.update(genres_str.split('|'))
        return sorted(list(all_genres))
    
    def predict(self, movie_id: int, n: int = 10):
        try:
            movie_idx = self.movies[self.movies['movieId'] == movie_id].index[0]
        except IndexError:
            return self.get_top_movies(n)
        
        sim_scores = self.similarity_matrix[movie_idx]
        similar_indices = sim_scores.argsort()[::-1][1:n+1]
        
        recommendations = self.movies.iloc[similar_indices]
        movie_ratings = self.ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
        
        result = recommendations.merge(
            movie_ratings, left_on='movieId', right_index=True, how='left'
        ).rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})
        
        return result

recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    print("\n🚀 Démarrage de MyTflix API...")
    print("📚 Chargement du modèle ML...")
    
    model_path = 'recommender_model.pkl'
    
    if not Path(model_path).exists():
        print("🔄 Entraînement du modèle (première utilisation)...")
        recommender = MovieRecommender()
        recommender.train()
        recommender.save(model_path)
        print("✅ Modèle entraîné et sauvegardé!")
    else:
        recommender = MovieRecommender.load(model_path)
        print("✅ Modèle chargé depuis le fichier!")
    
    print("✅ API prête à recevoir des requêtes\n")
    
    yield
    
    print("\n🛑 Arrêt de l'API MyTflix...")

app = FastAPI(
    title="MyTflix API",
    description="API de recommandation de films avec Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

class MovieResponse(BaseModel):
    movieId: int
    title: str
    genres: str
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10

class GenreRecommendationRequest(BaseModel):
    genres: List[str]
    n_recommendations: int = 15

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyTflix API</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                background: #0b0b0b;
                color: #fff;
                min-height: 100vh;
                padding: 40px 20px;
            }
            .container { max-width: 1000px; margin: 0 auto; }
            header {
                background: linear-gradient(180deg, rgba(229, 9, 20, 0.3) 0%, rgba(11, 11, 11, 0) 100%);
                padding: 60px 30px;
                border-bottom: 2px solid #e50914;
                margin-bottom: 40px;
                text-align: center;
            }
            h1 {
                font-size: 48px;
                font-weight: 900;
                letter-spacing: 2px;
                margin-bottom: 10px;
                color: #e50914;
            }
            .subtitle { color: #bbb; font-size: 16px; margin-bottom: 30px; }
            .card {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(229, 9, 20, 0.2);
                padding: 30px;
                border-radius: 8px;
                margin-bottom: 30px;
            }
            .card h2 { color: #e50914; margin-bottom: 20px; }
            .endpoint {
                background: rgba(229, 9, 20, 0.1);
                border-left: 4px solid #e50914;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }
            .method { font-weight: bold; color: #f5a623; }
            .docs-link {
                display: inline-block;
                background: linear-gradient(90deg, #e50914 0%, #f5a623 100%);
                color: white;
                padding: 12px 30px;
                border-radius: 4px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 20px;
            }
            .docs-link:hover { transform: scale(1.05); }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>MyTflix API</h1>
                <p class="subtitle">Système de recommandation de films avec Machine Learning</p>
            </header>
            
            <div class="card">
                <h2>Bienvenue sur MyTflix!</h2>
                <p>API FastAPI pour la recommandation de films utilisant des modèles Machine Learning entraînés.</p>
                
                <h3 style="margin-top: 30px; margin-bottom: 20px; color: #f5a623;">Endpoints disponibles:</h3>
                
                <div class="endpoint">
                    <div class="method">GET</div> /health - Vérifier l'état de l'API
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /top-films - Récupérer les meilleurs films
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /films/genre/{genre} - Films par genre
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /utilisateur/{user_id}/films - Films d'un utilisateur
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div> /recommandations - Recommandations personnalisées
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div> /recommandations/genres - Recommandations par genres
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /stats - Statistiques globales
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /genres - Liste des genres
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /utilisateurs/count - Nombre d'utilisateurs
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div> /recommend/{movie_title} - Recommandations par titre
                </div>
                
                <a href="/docs" class="docs-link">📚 Documentation complète (Swagger)</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    if recommender is None:
        return {"status": "initializing", "message": "Modèle en cours de chargement"}
    return {
        "status": "healthy",
        "message": "API opérationnelle",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/top-films", response_model=List[MovieResponse])
async def get_top_films(n: int = 20, skip: int = 0):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        top_movies = recommender.get_top_movies(n=n+skip)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in top_movies.iloc[skip:skip+n].iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/films/genre/{genre}", response_model=List[MovieResponse])
async def get_films_by_genre(genre: str, n: int = 20):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        movies = recommender.get_movies_by_genre(genre, n=n)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0))
            )
            for _, m in movies.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/utilisateur/{user_id}/films", response_model=List[MovieResponse])
async def get_user_films(user_id: int, n: int = 50):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        user_ratings = recommender.get_user_ratings(user_id, n=n)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('rating', 0))
            )
            for _, m in user_ratings.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommandations", response_model=List[MovieResponse])
async def get_recommendations(request: RecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        recs = recommender.get_recommendations_by_ratings(request.user_id, n=request.n_recommendations)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in recs.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommandations/genres", response_model=List[MovieResponse])
async def get_recommendations_by_genres(request: GenreRecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        recs = recommender.recommend_by_multiple_genres(request.genres, n=request.n_recommendations)
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in recs.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
async def get_stats():
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        return {
            "total_films": int(len(recommender.movies)),
            "total_ratings": int(len(recommender.ratings)),
            "total_users": int(recommender.ratings['userId'].nunique()),
            "avg_rating": float(recommender.ratings['rating'].mean()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/genres")
async def get_all_genres():
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        genres = recommender.get_all_genres()
        return {"genres": genres, "total": len(genres)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/utilisateurs/count")
async def get_users_count():
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        count = int(recommender.ratings['userId'].nunique())
        return {
            "total_users": count,
            "available_ids": list(recommender.ratings['userId'].unique()[:100])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommend/{movie_title}", response_model=List[MovieResponse])
async def recommend_by_title(movie_title: str, n: int = 10):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        matching_movies = recommender.movies[
            recommender.movies['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if matching_movies.empty:
            raise HTTPException(status_code=404, detail=f"Film non trouvé: {movie_title}")
        
        movie_id = int(matching_movies.iloc[0]['movieId'])
        recs = recommender.predict(movie_id, n=n)
        
        return [
            MovieResponse(
                movieId=int(m['movieId']),
                title=m['title'],
                genres=m['genres'],
                avg_rating=float(m.get('avg_rating', 0)),
                rating_count=int(m.get('rating_count', 0))
            )
            for _, m in recs.iterrows()
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
