# Setup instructions for MyTflix API

## 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 2. Préparer les données

Assurez-vous que les fichiers CSV sont présents:
- `movies.csv` - Liste des films
- `ratings.csv` - Évaluations des utilisateurs  
- `tags.csv` - Tags des films

Si vous n'avez pas les fichiers, exécutez:
```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)

# Créer movies.csv
movies_data = {
    'movieId': range(1, 101),
    'title': [f'Movie {i}' for i in range(1, 101)],
    'genres': ['Action|Adventure', 'Comedy|Drama', 'Action|Sci-Fi', 'Drama|Romance', 'Horror|Thriller'] * 20
}
movies = pd.DataFrame(movies_data)
movies.to_csv('movies.csv', index=False)

# Créer ratings.csv
ratings_data = {
    'userId': np.random.randint(1, 51, 500),
    'movieId': np.random.randint(1, 101, 500),
    'rating': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 500),
    'timestamp': np.random.randint(1000000000, 1200000000, 500)
}
ratings = pd.DataFrame(ratings_data)
ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
ratings.to_csv('ratings.csv', index=False)

# Créer tags.csv
tags_data = {
    'userId': np.random.randint(1, 51, 200),
    'movieId': np.random.randint(1, 101, 200),
    'tag': ['great', 'boring', 'funny', 'scary', 'amazing'] * 40,
    'timestamp': np.random.randint(1000000000, 1200000000, 200)
}
tags = pd.DataFrame(tags_data)
tags.to_csv('tags.csv', index=False)

print("✅ CSV files created!")
EOF
```

## 3. Lancer l'application

```bash
python app.py
```

L'API sera accessible à:
- **http://localhost:8000** - Page d'accueil
- **http://localhost:8000/docs** - Documentation Swagger

## 4. Utiliser l'API

### Exemple 1: Obtenir les meilleurs films
```bash
curl "http://localhost:8000/top-films?n=10"
```

### Exemple 2: Obtenir les recommandations
```bash
curl -X POST "http://localhost:8000/recommandations" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n_recommendations": 5}'
```

### Exemple 3: Films similaires
```bash
curl "http://localhost:8000/recommend/Movie%201?n=5"
```

## Fichiers générés

- `recommender_model.pkl` - Modèle ML sauvegardé (généré au premier lancement)
- `movies.csv` - Base de données des films
- `ratings.csv` - Base de données des évaluations
- `tags.csv` - Base de données des tags
