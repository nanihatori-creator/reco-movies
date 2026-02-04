# MyTflix - API FastAPI de Recommandation de Films

API Backend FastAPI avec Machine Learning pour recommander des films.

## Fonctionnalités

- **Modèle ML entraîné et sauvegardé** en pickle (.pkl)
- **Chargement automatique** du modèle au démarrage
- **Recommandations basées sur**:
  - Genres similaires
  - Évaluations d'utilisateurs
  - Titres de films
- **API REST complète** avec Swagger docs
- **Interface web** HTML intégrée

## Prérequis

- Python 3.10+
- Fichiers CSV: `movies.csv`, `ratings.csv`, `tags.csv`

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python app.py
```

L'API sera accessible sur:
- **API**: `http://localhost:8000`
- **Documentation Swagger**: `http://localhost:8000/docs`

## Endpoints Principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil |
| GET | `/health` | État de l'API |
| GET | `/top-films` | Meilleurs films |
| GET | `/films/genre/{genre}` | Films par genre |
| GET | `/utilisateur/{user_id}/films` | Films d'un utilisateur |
| POST | `/recommandations` | Recommandations personnalisées |
| POST | `/recommandations/genres` | Recommandations par genres |
| GET | `/stats` | Statistiques |
| GET | `/genres` | Liste des genres |
| GET | `/recommend/{movie_title}` | Films similaires |

## Architecture

- **MovieRecommender**: Classe principale pour le ML
- **Modèle sauvegardé**: `recommender_model.pkl`
- **Entraînement**: TF-IDF + Similarité cosinus
- **Lifespan**: Gestion du cycle de vie FastAPI

## Exemple d'utilisation

```bash
# Recommandations pour utilisateur 1
curl -X POST "http://localhost:8000/recommandations" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 10}'

# Films similaires à "Inception"
curl "http://localhost:8000/recommend/Inception?n=5"

# Top 20 films
curl "http://localhost:8000/top-films?n=20"
```

## Déploiement Azure

L'application est prête pour Azure App Service avec Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## License

MIT
