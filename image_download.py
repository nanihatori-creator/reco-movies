import requests
import os
from pathlib import Path

class PosterManager:
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
    POSTER_DIR = 'posters'
    
    @classmethod
    def init(cls):
        Path(cls.POSTER_DIR).mkdir(exist_ok=True)
    
    @classmethod
    def get_or_download_poster(cls, movie_title: str, movie_id: int):
        if not cls.TMDB_API_KEY:
            return None
        
        poster_path = Path(cls.POSTER_DIR) / f"{movie_id}.jpg"
        
        if poster_path.exists():
            return str(poster_path)
        
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': cls.TMDB_API_KEY,
                'query': movie_title
            }
            response = requests.get(search_url, params=params, timeout=5)
            results = response.json().get('results', [])
            
            if results:
                poster_path_tmdb = results[0].get('poster_path')
                if poster_path_tmdb:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path_tmdb}"
                    img_response = requests.get(poster_url, timeout=5)
                    
                    if img_response.status_code == 200:
                        with open(poster_path, 'wb') as f:
                            f.write(img_response.content)
                        return str(poster_path)
        except Exception as e:
            print(f"Erreur telechargement affiche: {e}")
        
        return None

def get_or_download_poster(movie_title: str, movie_id: int):
    return PosterManager.get_or_download_poster(movie_title, movie_id)
