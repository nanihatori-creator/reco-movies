import os
import requests
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
TMDB_API_KEY = "e5c934fe24429749beb4d1f4724bb2ee"
POSTER_DIR = "posters"
IMAGE_SIZE = "w500"  # w200, w500, original

MOVIES_PATH = "movies.csv"
LINKS_PATH = "links.csv"

# =========================
# CREATE FOLDER
# =========================
os.makedirs(POSTER_DIR, exist_ok=True)

# Utility functions: do not run downloads on import

def search_tmdb(title, year=None):
    """Search TMDB for a movie by title (and optional year). Returns TMDB id or None."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year is not None:
        params["year"] = int(year)

    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0].get("id")
    except Exception:
        return None
    return None


def get_poster_url_from_tmdb(tmdb_id):
    """Return full poster URL for a given tmdb movie id, or None."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
        params = {"api_key": TMDB_API_KEY}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            poster_path = r.json().get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/{IMAGE_SIZE}{poster_path}"
    except Exception:
        return None
    return None


def download_image(url, dest_path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        return False
    return False


def extract_year_from_title(title):
    import re
    m = re.search(r"\((\d{4})\)", str(title))
    if m:
        return int(m.group(1))
    return None


def get_or_download_poster(movie_id, title):
    """Return local poster path for a movie. If missing, try to download from TMDB.

    Args:
        movie_id: int or str - id from movies.csv
        title: movie title (used for search)

    Returns: local filepath string or None
    """
    try:
        img_path = os.path.join(POSTER_DIR, f"{int(movie_id)}.jpg")
    except Exception:
        img_path = os.path.join(POSTER_DIR, f"{movie_id}.jpg")

    if os.path.exists(img_path):
        return img_path

    # Attempt TMDB search using title and year
    year = extract_year_from_title(title)
    tmdb_id = search_tmdb(title, year=year)
    if not tmdb_id:
        tmdb_id = search_tmdb(title)  # try without year
    if not tmdb_id:
        return None

    poster_url = get_poster_url_from_tmdb(tmdb_id)
    if not poster_url:
        return None

    success = download_image(poster_url, img_path)
    if success:
        return img_path
    return None
