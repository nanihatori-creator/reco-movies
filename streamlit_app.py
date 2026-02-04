import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from image_download import PosterManager

st.set_page_config(
    page_title="MyTflix",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"

st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #0b0b0b 0%, #1a1a1a 100%);
        color: #fff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    
    .netflix-header {
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.8) 0%, rgba(245, 166, 35, 0.6) 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3);
        border: 1px solid rgba(229, 9, 20, 0.5);
        text-align: center;
    }
    
    .netflix-header h1 {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 2px;
        margin-bottom: 10px;
        color: #fff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .netflix-header p {
        font-size: 16px;
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(229, 9, 20, 0.3);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .card:hover {
        border-color: rgba(229, 9, 20, 0.6);
        box-shadow: 0 8px 24px rgba(229, 9, 20, 0.2);
        transition: all 0.3s ease;
    }
    
    .movie-card {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(245, 166, 35, 0.05) 100%);
        border-left: 4px solid #e50914;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 6px;
    }
    
    .movie-title {
        color: #e50914;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 8px;
    }
    
    .movie-genres {
        color: #f5a623;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .movie-rating {
        color: #00d084;
        font-weight: bold;
        font-size: 16px;
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.2) 0%, rgba(245, 166, 35, 0.1) 100%);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: bold;
        color: #e50914;
        margin-bottom: 10px;
    }
    
    .stat-label {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.8);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="netflix-header">
    <h1>MyTflix</h1>
    <p>Systeme de recommandation de films avec Machine Learning</p>
</div>
""", unsafe_allow_html=True)

try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    api_status = health_response.json()
    is_healthy = api_status.get('status') == 'healthy'
except:
    is_healthy = False

if not is_healthy:
    st.error("L'API n'est pas disponible. Demarrez le serveur: python app.py")
    st.stop()

st.success("API connectee et operationnelle")

col1, col2, col3 = st.columns(3)

try:
    stats = requests.get(f"{API_BASE_URL}/stats", timeout=5).json()
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['total_films']}</div>
            <div class="stat-label">Films disponibles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['total_users']}</div>
            <div class="stat-label">Utilisateurs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{stats['avg_rating']:.1f}/5</div>
            <div class="stat-label">Note moyenne</div>
        </div>
        """, unsafe_allow_html=True)
except:
    st.error("Erreur chargement statistiques")

tabs = st.tabs([
    "Recommandations",
    "Meilleurs Films",
    "Par Genre",
    "Statistiques",
    "Profil Utilisateur"
])

# Initialize poster cache
PosterManager.init()

PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Poster"

with tabs[0]:
    st.markdown("## Recommandations Personnalisees")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.number_input("ID Utilisateur", min_value=1, value=1, step=1)
    with col2:
        n_recs = st.number_input("Nombre", min_value=1, max_value=50, value=10, step=1)
    
    if st.button("Obtenir Recommandations"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/recommandations",
                json={"user_id": user_id, "n_recommendations": n_recs},
                timeout=10
            )
            if response.status_code == 200:
                movies = response.json()
                if movies:
                    cols = st.columns(4)
                    for i, movie in enumerate(movies):
                        col = cols[i % 4]
                        with col:
                            poster = PosterManager.get_or_download_poster(movie.get('title', ''), int(movie.get('movieId', 0)))
                            img_src = poster if poster else PLACEHOLDER
                            st.image(img_src, use_column_width=True)
                            st.markdown(f"""
                            <div style='padding-top:6px;'>
                                <div style='font-weight:700;color:#fff'>{movie['title']}</div>
                                <div style='color:#f5a623;font-size:12px'>{movie['genres']}</div>
                                <div style='color:#00d084;font-weight:600'>★ {movie.get('avg_rating', 'N/A')} ({movie.get('rating_count', 0)} votes)</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Aucune recommandation disponible")
            else:
                st.error(f"Erreur: {response.json().get('detail', 'Erreur inconnue')}")
        except Exception as e:
            st.error(f"Erreur de connexion: {str(e)}")

with tabs[1]:
    st.markdown("## Meilleurs Films")
    
    n_top = st.slider("Nombre de films", 5, 50, 20)
    
    try:
        response = requests.get(f"{API_BASE_URL}/top-films?n={n_top}", timeout=10)
        if response.status_code == 200:
            movies = response.json()
            cols = st.columns(4)
            for i, movie in enumerate(movies):
                col = cols[i % 4]
                with col:
                    poster = PosterManager.get_or_download_poster(movie.get('title', ''), int(movie.get('movieId', 0)))
                    img_src = poster if poster else PLACEHOLDER
                    st.image(img_src, use_column_width=True)
                    st.markdown(f"""
                    <div style='padding-top:6px;'>
                        <div style='font-weight:700;color:#fff'>{i+1}. {movie['title']}</div>
                        <div style='color:#f5a623;font-size:12px'>{movie['genres']}</div>
                        <div style='color:#00d084;font-weight:600'>★ {movie.get('avg_rating', 'N/A')}/5 ({movie.get('rating_count', 0)} votes)</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

with tabs[2]:
    st.markdown("## Films par Genre")
    
    try:
        genres_response = requests.get(f"{API_BASE_URL}/genres", timeout=10)
        genres_list = genres_response.json()['genres']
        
        selected_genre = st.selectbox("Selectionner un genre", genres_list)
        n_genre = st.slider("Nombre de films", 5, 50, 15, key="genre_slider")
        
        if st.button("Charger Films"):
            response = requests.get(
                f"{API_BASE_URL}/films/genre/{selected_genre}?n={n_genre}",
                timeout=10
            )
            if response.status_code == 200:
                movies = response.json()
                cols = st.columns(4)
                for i, movie in enumerate(movies):
                    col = cols[i % 4]
                    with col:
                        poster = PosterManager.get_or_download_poster(movie.get('title', ''), int(movie.get('movieId', 0)))
                        img_src = poster if poster else PLACEHOLDER
                        st.image(img_src, use_column_width=True)
                        st.markdown(f"""
                        <div style='padding-top:6px;'>
                            <div style='font-weight:700;color:#fff'>{movie['title']}</div>
                            <div style='color:#f5a623;font-size:12px'>{movie['genres']}</div>
                            <div style='color:#00d084;font-weight:600'>★ {movie.get('avg_rating', 'N/A')}</div>
                        </div>
                        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

with tabs[3]:
    st.markdown("## Statistiques Globales")
    
    try:
        stats = requests.get(f"{API_BASE_URL}/stats", timeout=10).json()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Films", stats['total_films'])
        with col2:
            st.metric("Utilisateurs", stats['total_users'])
        with col3:
            st.metric("Evaluations", stats['total_ratings'])
        with col4:
            st.metric("Note Moyenne", f"{stats['avg_rating']:.2f}/5")
        
        st.subheader("Donnees Globales")
        stats_data = {
            "Metrique": ["Films", "Utilisateurs", "Evaluations", "Note Moyenne"],
            "Valeur": [
                stats['total_films'],
                stats['total_users'],
                stats['total_ratings'],
                f"{stats['avg_rating']:.2f}"
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

with tabs[4]:
    st.markdown("## Profil Utilisateur")
    
    user_id = st.number_input("Entrez votre ID utilisateur", min_value=1, value=1, step=1, key="profile_user_id")
    
    if st.button("Voir mon Profil"):
        try:
            response = requests.get(
                f"{API_BASE_URL}/utilisateur/{user_id}/films?n=20",
                timeout=10
            )
            if response.status_code == 200:
                movies = response.json()
                
                if movies:
                    st.subheader(f"Films evalues par l'utilisateur {user_id}")
                    cols = st.columns(3)
                    for i, movie in enumerate(movies):
                        col = cols[i % 3]
                        with col:
                            poster = PosterManager.get_or_download_poster(movie.get('title', ''), int(movie.get('movieId', 0)))
                            img_src = poster if poster else PLACEHOLDER
                            st.image(img_src, use_column_width=True)
                            st.write(f"**{movie['title']}**")
                            st.caption(movie['genres'])
                            st.metric("Note", f"{movie.get('avg_rating', 'N/A')}/5")
                else:
                    st.info("Cet utilisateur n'a pas d'evaluations")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

st.markdown("""
<hr style="border: 1px solid rgba(229, 9, 20, 0.3); margin-top: 40px; margin-bottom: 20px;">
<div style="text-align: center; color: rgba(255, 255, 255, 0.6); font-size: 12px;">
    <p>MyTflix 2026 | Systeme de recommandation intelligent | Propulse par FastAPI & Streamlit</p>
</div>
""", unsafe_allow_html=True)
