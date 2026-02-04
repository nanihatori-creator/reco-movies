from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os
from collections import Counter

app = FastAPI()

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
tags_df = pd.read_csv('tags.csv')

def prepare_data():
    merged_df = ratings_df.merge(movies_df, on='movieId', how='left')
    merged_df = merged_df.merge(tags_df[['movieId', 'tag']].drop_duplicates(), 
                                on='movieId', how='left')
    
    genre_list = merged_df['genres'].fillna('').apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(genre_list)
    genre_df = pd.DataFrame(genre_encoded, columns=[f'genre_{g}' for g in mlb.classes_])
    
    X = pd.concat([
        merged_df[['userId', 'movieId']],
        genre_df.reset_index(drop=True)
    ], axis=1)
    
    y = merged_df['rating']
    
    return X, y, merged_df, mlb

X, y, merged_df, mlb = prepare_data()

sample_size = min(50000, len(X))
indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[indices]
y_sample = y.iloc[indices]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sample, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=5)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, y_pred_gb)
gb_rmse = np.sqrt(gb_mse)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_r2 = r2_score(y_test, y_pred_gb)

stats_mean = float(ratings_df['rating'].mean())
stats_std = float(ratings_df['rating'].std())
stats_median = float(ratings_df['rating'].median())
stats_min = float(ratings_df['rating'].min())
stats_max = float(ratings_df['rating'].max())

@app.get("/", response_class=HTMLResponse)
def get_home():
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>CinemaML - Prédiction de Notes</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: #0b0b0b;
            color: #fff;
            min-height: 100vh;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(180deg, rgba(229, 9, 20, 0.3) 0%, rgba(11, 11, 11, 0) 100%);
            padding: 60px 30px;
            border-bottom: 2px solid #e50914;
            margin-bottom: 40px;
            text-align: center;
        }}
        h1 {{
            font-size: 48px;
            font-weight: 900;
            letter-spacing: 2px;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #e50914 0%, #f5a623 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{ color: #bbb; font-size: 16px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(229, 9, 20, 0.2);
            padding: 30px;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .card:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: #e50914;
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(229, 9, 20, 0.3);
        }}
        .card h2 {{
            color: #e50914;
            margin-bottom: 25px;
            font-size: 18px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            align-items: center;
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #999; font-size: 13px; }}
        .metric-value {{ 
            font-weight: 700;
            color: #e50914;
            font-size: 18px;
        }}
        .stats-card {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
        }}
        .stat-item {{
            background: rgba(229, 9, 20, 0.1);
            border: 1px solid rgba(229, 9, 20, 0.3);
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .stat-item:hover {{
            background: rgba(229, 9, 20, 0.2);
            border-color: #e50914;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: 900;
            color: #e50914;
        }}
        .stat-label {{
            font-size: 12px;
            color: #999;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .prediction-form {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(229, 9, 20, 0.2);
            padding: 40px;
            border-radius: 8px;
            margin-bottom: 40px;
        }}
        .prediction-form h2 {{
            color: #e50914;
            margin-bottom: 30px;
            font-size: 24px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .form-group {{ margin-bottom: 20px; }}
        label {{
            display: block;
            margin-bottom: 8px;
            color: #fff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}
        input, select {{
            width: 100%;
            padding: 14px;
            border: 1px solid rgba(229, 9, 20, 0.3);
            border-radius: 4px;
            font-size: 14px;
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            transition: all 0.3s ease;
        }}
        input:focus, select:focus {{
            outline: none;
            border-color: #e50914;
            box-shadow: 0 0 10px rgba(229, 9, 20, 0.3);
            background: rgba(255, 255, 255, 0.08);
        }}
        input::placeholder {{ color: #666; }}
        .form-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        button {{
            background: linear-gradient(90deg, #e50914 0%, #f5a623 100%);
            color: white;
            padding: 14px 40px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 700;
            width: 100%;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }}
        button:hover {{
            transform: scale(1.02);
            box-shadow: 0 8px 20px rgba(229, 9, 20, 0.4);
        }}
        button:active {{ transform: scale(0.98); }}
        .result {{
            background: rgba(229, 9, 20, 0.1);
            border: 2px solid #e50914;
            border-left: 6px solid #e50914;
            padding: 25px;
            border-radius: 4px;
            display: none;
            margin-top: 30px;
        }}
        .result.show {{ display: block; animation: slideIn 0.3s ease; }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .result h3 {{
            color: #e50914;
            margin-bottom: 15px;
            text-transform: uppercase;
            font-size: 14px;
            letter-spacing: 1px;
        }}
        .result-value {{
            font-size: 48px;
            font-weight: 900;
            color: #e50914;
            margin-bottom: 20px;
        }}
        .stats-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }}
        .stat {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(229, 9, 20, 0.3);
            padding: 15px;
            border-radius: 4px;
        }}
        .stat strong {{ color: #e50914; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 30px; margin-bottom: 40px; }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(229, 9, 20, 0.2);
            padding: 30px;
            border-radius: 8px;
        }}
        .chart-container h3 {{
            color: #e50914;
            margin-bottom: 25px;
            text-transform: uppercase;
            font-size: 14px;
            letter-spacing: 1px;
        }}
        @media (max-width: 768px) {{
            h1 {{ font-size: 32px; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
            .stats-card {{ grid-template-columns: repeat(2, 1fr); }}
            .form-row {{ grid-template-columns: 1fr; }}
            .result-value {{ font-size: 32px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>CinemaML</h1>
            <p class="subtitle">Prédiction Intelligente de Notes de Films avec Machine Learning</p>
        </header>
        
        <div class="card" style="margin-bottom: 40px; border-color: rgba(229, 9, 20, 0.5);">
            <h2>Statistiques Globales</h2>
            <div class="stats-card">
                <div class="stat-item">
                    <div class="stat-value">{stats_mean:.2f}</div>
                    <div class="stat-label">Moyenne</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats_std:.2f}</div>
                    <div class="stat-label">Écart-type</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats_median:.2f}</div>
                    <div class="stat-label">Médiane</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats_min:.2f}</div>
                    <div class="stat-label">Minimum</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats_max:.2f}</div>
                    <div class="stat-label">Maximum</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Random Forest</h2>
                <div class="metric">
                    <span class="metric-label">RMSE</span>
                    <span class="metric-value">{rf_rmse:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">MAE</span>
                    <span class="metric-value">{rf_mae:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">R² Score</span>
                    <span class="metric-value">{rf_r2:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Statut</span>
                    <span class="metric-value">Actif</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Gradient Boosting</h2>
                <div class="metric">
                    <span class="metric-label">RMSE</span>
                    <span class="metric-value">{gb_rmse:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">MAE</span>
                    <span class="metric-value">{gb_mae:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">R² Score</span>
                    <span class="metric-value">{gb_r2:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Statut</span>
                    <span class="metric-value">Actif</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Base de Données</h2>
                <div class="metric">
                    <span class="metric-label">Films</span>
                    <span class="metric-value">{len(movies_df)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Utilisateurs</span>
                    <span class="metric-value">{ratings_df['userId'].nunique()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Évaluations</span>
                    <span class="metric-value">{len(ratings_df)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Entraînement</span>
                    <span class="metric-value">{len(X_test)} tests</span>
                </div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Distribution des Notes</h3>
                <canvas id="histogramChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Genres Populaires</h3>
                <canvas id="genreChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Notes au Fil du Temps</h3>
                <canvas id="areaChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Performance des Modèles</h3>
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
        
        <div class="prediction-form">
            <h2>Prédire une Note</h2>
            <div class="form-row">
                <div class="form-group">
                    <label>ID Utilisateur</label>
                    <input type="number" id="userId" placeholder="Ex: 1" min="1" max="{ratings_df['userId'].max()}">
                </div>
                <div class="form-group">
                    <label>Film</label>
                    <select id="movieId">
                        <option value="">Sélectionner un film...</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label>Genres</label>
                <input type="text" id="genres" placeholder="Ex: Action|Adventure|Sci-Fi" readonly>
            </div>
            <button onclick="predict()">Prédire la Note</button>
            
            <div class="result" id="result">
                <h3>Résultat de la Prédiction</h3>
                <div class="result-value" id="predictionValue"></div>
                <div class="stats-row">
                    <div class="stat">
                        <strong>Random Forest</strong><br>
                        <span id="rfPred">-</span> / 5
                    </div>
                    <div class="stat">
                        <strong>Gradient Boosting</strong><br>
                        <span id="gbPred">-</span> / 5
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const movieSelect = document.getElementById('movieId');
        const genreInput = document.getElementById('genres');
        
        fetch('/api/movies')
            .then(r => r.json())
            .then(data => {{
                data.movies.forEach(m => {{
                    const option = document.createElement('option');
                    option.value = m.movieId;
                    option.textContent = m.title;
                    movieSelect.appendChild(option);
                }});
            }});
        
        movieSelect.addEventListener('change', function() {{
            fetch(`/api/movie/${{this.value}}`)
                .then(r => r.json())
                .then(data => {{
                    genreInput.value = data.genres || '';
                }});
        }});
        
        function predict() {{
            const userId = document.getElementById('userId').value;
            const movieId = document.getElementById('movieId').value;
            
            if (!userId || !movieId) {{
                alert('Remplissez tous les champs');
                return;
            }}
            
            fetch(`/api/predict?userId=${{userId}}&movieId=${{movieId}}`)
                .then(r => r.json())
                .then(data => {{
                    if (data.error) {{
                        alert('Erreur: ' + data.error);
                        return;
                    }}
                    document.getElementById('predictionValue').textContent = data.average.toFixed(2);
                    document.getElementById('rfPred').textContent = data.rf_prediction.toFixed(2);
                    document.getElementById('gbPred').textContent = data.gb_prediction.toFixed(2);
                    document.getElementById('result').classList.add('show');
                }})
                .catch(e => alert('Erreur réseau: ' + e));
        }}
        
        function initCharts() {{
            fetch('/api/statistics')
                .then(r => r.json())
                .then(data => {{
                    initHistogram(data.rating_distribution);
                    initGenreChart(data.genre_distribution);
                    initAreaChart(data.time_series);
                    initPerformanceChart(data.models);
                }});
        }}
        
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: true,
            plugins: {{
                legend: {{
                    labels: {{
                        color: '#fff',
                        font: {{ weight: 'bold' }}
                    }}
                }}
            }},
            scales: {{
                x: {{
                    ticks: {{ color: '#999' }},
                    grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}
                }},
                y: {{
                    ticks: {{ color: '#999' }},
                    grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}
                }}
            }}
        }};
        
        function initHistogram(data) {{
            const ctx = document.getElementById('histogramChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.bins,
                    datasets: [{{
                        label: 'Nombre de Notes',
                        data: data.counts,
                        backgroundColor: '#e50914',
                        borderColor: 'rgba(229, 9, 20, 0.5)',
                        borderWidth: 1
                    }}]
                }},
                options: {{...chartOptions, plugins: {{...chartOptions.plugins, legend: {{ display: false }}}}}}
            }});
        }}
        
        function initGenreChart(data) {{
            const ctx = document.getElementById('genreChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.genres,
                    datasets: [{{
                        label: 'Nombre de Films',
                        data: data.counts,
                        backgroundColor: '#f5a623',
                        borderColor: 'rgba(245, 166, 35, 0.5)',
                        borderWidth: 1
                    }}]
                }},
                options: {{...chartOptions, indexAxis: 'y', plugins: {{...chartOptions.plugins, legend: {{ display: false }}}}}}
            }});
        }}
        
        function initAreaChart(data) {{
            const ctx = document.getElementById('areaChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: data.timestamps,
                    datasets: [{{
                        label: 'Notes Moyennes',
                        data: data.values,
                        borderColor: '#e50914',
                        backgroundColor: 'rgba(229, 9, 20, 0.2)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2
                    }}]
                }},
                options: {{...chartOptions, scales: {{...chartOptions.scales, y: {{...chartOptions.scales.y, beginAtZero: true, max: 5}}}}}}
            }});
        }}
        
        function initPerformanceChart(data) {{
            const ctx = document.getElementById('performanceChart').getContext('2d');
            new Chart(ctx, {{
                type: 'radar',
                data: {{
                    labels: ['RMSE', 'MAE', 'R² Score'],
                    datasets: [
                        {{
                            label: 'Random Forest',
                            data: [data.rf.rmse, data.rf.mae, data.rf.r2],
                            borderColor: '#e50914',
                            backgroundColor: 'rgba(229, 9, 20, 0.2)',
                            pointBackgroundColor: '#e50914',
                            pointBorderColor: '#fff'
                        }},
                        {{
                            label: 'Gradient Boosting',
                            data: [data.gb.rmse, data.gb.mae, data.gb.r2],
                            borderColor: '#f5a623',
                            backgroundColor: 'rgba(245, 166, 35, 0.2)',
                            pointBackgroundColor: '#f5a623',
                            pointBorderColor: '#fff'
                        }}
                    ]
                }},
                options: {{...chartOptions}}
            }});
        }}
        
        initCharts();
    </script>
</body>
</html>
    """

@app.get("/api/movies")
def get_movies_list():
    return {"movies": movies_df.head(100).to_dict('records')}

@app.get("/api/movie/{movie_id}")
def get_movie(movie_id: int):
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        return {"error": "Film non trouvé"}
    return movie.iloc[0].to_dict()

@app.get("/api/predict")
def predict_rating(userId: int, movieId: int):
    try:
        movie = movies_df[movies_df['movieId'] == movieId]
        if movie.empty:
            return {"error": "Film non trouvé"}
        
        genres = movie.iloc[0]['genres'].split('|')
        genre_encoded = mlb.transform([genres])
        
        X_new = np.concatenate([[userId, movieId], genre_encoded[0]])
        X_new_scaled = scaler.transform([X_new])
        
        rf_pred = rf_model.predict(X_new_scaled)[0]
        gb_pred = gb_model.predict(X_new_scaled)[0]
        avg_pred = (rf_pred + gb_pred) / 2
        
        rf_pred = max(0.5, min(5.0, rf_pred))
        gb_pred = max(0.5, min(5.0, gb_pred))
        avg_pred = max(0.5, min(5.0, avg_pred))
        
        return {
            "userId": userId,
            "movieId": movieId,
            "title": movie.iloc[0]['title'],
            "rf_prediction": float(rf_pred),
            "gb_prediction": float(gb_pred),
            "average": float(avg_pred)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/statistics")
def get_statistics():
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    
    rating_dist = {
        "bins": [f"{r}" for r in rating_counts.index],
        "counts": rating_counts.values.tolist()
    }
    
    genre_movie_count = {}
    for genres in movies_df['genres'].fillna(''):
        for genre in genres.split('|'):
            genre_movie_count[genre] = genre_movie_count.get(genre, 0) + 1
    
    sorted_genres = sorted(genre_movie_count.items(), key=lambda x: x[1], reverse=True)[:10]
    genre_dist = {
        "genres": [g[0] for g in sorted_genres],
        "counts": [g[1] for g in sorted_genres]
    }
    
    ratings_sorted = ratings_df.sort_values('timestamp')
    chunk_size = len(ratings_sorted) // 20
    timestamps = []
    values = []
    
    for i in range(0, len(ratings_sorted), chunk_size):
        chunk = ratings_sorted.iloc[i:i+chunk_size]
        if len(chunk) > 0:
            timestamps.append(f"Pt {len(timestamps)+1}")
            values.append(float(chunk['rating'].mean()))
    
    time_series = {
        "timestamps": timestamps,
        "values": values
    }
    
    models = {
        "rf": {
            "rmse": float(rf_rmse),
            "mae": float(rf_mae),
            "r2": float(rf_r2)
        },
        "gb": {
            "rmse": float(gb_rmse),
            "mae": float(gb_mae),
            "r2": float(gb_r2)
        }
    }
    
    return {
        "rating_distribution": rating_dist,
        "genre_distribution": genre_dist,
        "time_series": time_series,
        "models": models
    }

@app.get("/api/metrics")
def get_metrics():
    return {
        "random_forest": {
            "rmse": float(rf_rmse),
            "mae": float(rf_mae),
            "r2": float(rf_r2)
        },
        "gradient_boosting": {
            "rmse": float(gb_rmse),
            "mae": float(gb_mae),
            "r2": float(gb_r2)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/", response_class=HTMLResponse)
def get_home():
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Prédiction de Notes - Machine Learning</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 30px; }}
        h1 {{ color: #1e3c72; margin-bottom: 5px; }}
        .subtitle {{ color: #666; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #2a5298; margin-bottom: 20px; font-size: 18px; border-bottom: 2px solid #2a5298; padding-bottom: 10px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
        .metric-label {{ color: #666; }}
        .metric-value {{ font-weight: bold; color: #1e3c72; }}
        .prediction-form {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .form-group {{ margin-bottom: 20px; }}
        label {{ display: block; margin-bottom: 8px; color: #333; font-weight: bold; }}
        input, select {{ width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }}
        .form-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
        button {{ background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%); color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; width: 100%; transition: 0.3s; }}
        button:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
        .result {{ background: #f0f8ff; border-left: 4px solid #2a5298; padding: 15px; border-radius: 5px; display: none; }}
        .result.show {{ display: block; }}
        .result h3 {{ color: #2a5298; margin-bottom: 10px; }}
        .result-value {{ font-size: 28px; font-weight: bold; color: #1e3c72; }}
        .stats-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px; }}
        .stat {{ background: white; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Système de Prédiction de Notes - Machine Learning</h1>
            <p class="subtitle">Prédiction basée sur les données de films et d'utilisateurs</p>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>Random Forest</h2>
                <div class="metric">
                    <span class="metric-label">RMSE</span>
                    <span class="metric-value">{rf_rmse:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">MAE</span>
                    <span class="metric-value">{rf_mae:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">R² Score</span>
                    <span class="metric-value">{rf_r2:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Modèle</span>
                    <span class="metric-value">Entraîné</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Gradient Boosting</h2>
                <div class="metric">
                    <span class="metric-label">RMSE</span>
                    <span class="metric-value">{gb_rmse:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">MAE</span>
                    <span class="metric-value">{gb_mae:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">R² Score</span>
                    <span class="metric-value">{gb_r2:.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Modèle</span>
                    <span class="metric-value">Entraîné</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Données</h2>
                <div class="metric">
                    <span class="metric-label">Total Films</span>
                    <span class="metric-value">{len(movies_df)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Utilisateurs</span>
                    <span class="metric-value">{ratings_df['userId'].nunique()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Notes</span>
                    <span class="metric-value">{len(ratings_df)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ensemble Test</span>
                    <span class="metric-value">{len(X_test)} samples</span>
                </div>
            </div>
        </div>
        
        <div class="prediction-form">
            <h2>Prédire une Note</h2>
            <div class="form-row">
                <div class="form-group">
                    <label>ID Utilisateur</label>
                    <input type="number" id="userId" placeholder="Ex: 1" min="1" max="{ratings_df['userId'].max()}">
                </div>
                <div class="form-group">
                    <label>Sélectionner un Film</label>
                    <select id="movieId">
                        <option value="">Choisir un film...</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label>Genres</label>
                <input type="text" id="genres" placeholder="Ex: Action|Adventure|Sci-Fi" readonly>
            </div>
            <button onclick="predict()">Prédire la Note</button>
            
            <div class="result" id="result">
                <h3>Prédiction</h3>
                <div class="result-value" id="predictionValue"></div>
                <div class="stats-row">
                    <div class="stat">
                        <strong>Random Forest:</strong><br>
                        <span id="rfPred">-</span> / 5
                    </div>
                    <div class="stat">
                        <strong>Gradient Boosting:</strong><br>
                        <span id="gbPred">-</span> / 5
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const movieSelect = document.getElementById('movieId');
        const genreInput = document.getElementById('genres');
        
        fetch('/api/movies')
            .then(r => r.json())
            .then(data => {{
                data.movies.forEach(m => {{
                    const option = document.createElement('option');
                    option.value = m.movieId;
                    option.textContent = m.title;
                    movieSelect.appendChild(option);
                }});
            }});
        
        movieSelect.addEventListener('change', function() {{
            fetch(`/api/movie/${{this.value}}`)
                .then(r => r.json())
                .then(data => {{
                    genreInput.value = data.genres;
                }});
        }});
        
        function predict() {{
            const userId = document.getElementById('userId').value;
            const movieId = document.getElementById('movieId').value;
            
            if (!userId || !movieId) {{
                alert('Remplissez tous les champs');
                return;
            }}
            
            fetch(`/api/predict?userId=${{userId}}&movieId=${{movieId}}`)
                .then(r => r.json())
                .then(data => {{
                    document.getElementById('predictionValue').textContent = data.average.toFixed(2);
                    document.getElementById('rfPred').textContent = data.rf_prediction.toFixed(2);
                    document.getElementById('gbPred').textContent = data.gb_prediction.toFixed(2);
                    document.getElementById('result').classList.add('show');
                }})
                .catch(e => alert('Erreur: ' + e));
        }}
    </script>
</body>
</html>
    """

@app.get("/api/movies")
def get_movies_list():
    return {"movies": movies_df.head(100).to_dict('records')}

@app.get("/api/movie/{movie_id}")
def get_movie(movie_id: int):
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        return {"error": "Film non trouvé"}
    return movie.iloc[0].to_dict()

@app.get("/api/predict")
def predict_rating(userId: int, movieId: int):
    try:
        movie = movies_df[movies_df['movieId'] == movieId]
        if movie.empty:
            return {"error": "Film non trouvé"}
        
        genres = movie.iloc[0]['genres'].split('|')
        genre_encoded = mlb.transform([genres])
        
        X_new = np.concatenate([[userId, movieId], genre_encoded[0]])
        X_new_scaled = scaler.transform([X_new])
        
        rf_pred = rf_model.predict(X_new_scaled)[0]
        gb_pred = gb_model.predict(X_new_scaled)[0]
        avg_pred = (rf_pred + gb_pred) / 2
        
        rf_pred = max(0.5, min(5.0, rf_pred))
        gb_pred = max(0.5, min(5.0, gb_pred))
        avg_pred = max(0.5, min(5.0, avg_pred))
        
        return {
            "userId": userId,
            "movieId": movieId,
            "title": movie.iloc[0]['title'],
            "rf_prediction": float(rf_pred),
            "gb_prediction": float(gb_pred),
            "average": float(avg_pred)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/metrics")
def get_metrics():
    return {
        "random_forest": {
            "rmse": float(rf_rmse),
            "mae": float(rf_mae),
            "r2": float(rf_r2)
        },
        "gradient_boosting": {
            "rmse": float(gb_rmse),
            "mae": float(gb_mae),
            "r2": float(gb_r2)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)