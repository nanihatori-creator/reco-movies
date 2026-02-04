# CinemaML - Movie Rating Prediction System

Système de prédiction de notes de films utilisant le Machine Learning avec une interface style Netflix.

## Fonctionnalités

- **Prédiction ML** : Deux modèles entraînés (Random Forest + Gradient Boosting)
- **Analyse des données** : Statistiques complètes et diagrammes interactifs
- **Interface Netflix** : Design moderne avec glassmorphism et animations
- **Visualisations** : Histogramme, graphiques en aires, radar des performances
- **API REST** : Endpoints pour prédictions et statistiques

## Technologies

- **Backend** : FastAPI + Uvicorn
- **ML** : Scikit-learn (Random Forest, Gradient Boosting)
- **Frontend** : HTML5 + CSS3 + Chart.js
- **Data** : Pandas + NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python app.py
```

Accédez à `http://localhost:8000`

## Structure des fichiers

- `app.py` - Application principale FastAPI
- `movies.csv` - Dataset des films
- `ratings.csv` - Dataset des évaluations
- `tags.csv` - Tags des films
- `links.csv` - Liens externes
- `requirements.txt` - Dépendances Python

## API Endpoints

- `GET /` - Interface Web
- `GET /api/movies` - Liste des films
- `GET /api/movie/{id}` - Détails d'un film
- `GET /api/predict` - Prédiction de note
- `GET /api/statistics` - Statistiques et données pour graphiques
- `GET /api/metrics` - Métriques des modèles

## Performance des Modèles

Les modèles sont entraînés sur un échantillon de 50,000 évaluations pour une rapidité optimale.

- **Random Forest** : 50 estimateurs, profondeur max 15
- **Gradient Boosting** : 50 estimateurs, profondeur max 5

## Design

Interface inspirée de Netflix avec :
- Palette de couleurs : Rouge (#e50914) et Orange (#f5a623)
- Fond sombre (#0b0b0b)
- Effet glassmorphism
- Animations fluides
- Design responsive
