Azure deployment notes

1) Choose the app to deploy
- If you want to deploy the FastAPI backend, keep `app.py` as the entrypoint.
- Do NOT use `streamlit_app.py` as an entrypoint for Azure Web App unless you intentionally want to serve Streamlit.

2) Remove the incorrect startup command in Azure (Deployment Center > Settings) if present:
- Remove: `uvicorn streamlit_app:app --host 0.0.0.0 --port 8000`

3) Use Gunicorn for production (recommended on Azure)
- If your FastAPI entrypoint file is `app.py`, use this startup command:

```
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

- Gunicorn will manage the worker processes and let Azure supply the port.

4) File paths and data
- `app.py` has been updated to use absolute paths (based on the repository root):

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
self.movies = pd.read_csv(BASE_DIR / 'movies.csv')
```

- Ensure your `movies.csv`, `ratings.csv`, `tags.csv` and other data files are present in the repository and deployed to the app service.

5) Optional: If you prefer to deploy the Streamlit UI instead of the API
- Rename `streamlit_app.py` to `main.py` and set the startup command to:

```
streamlit run main.py --server.runOnSave=false --server.port $PORT
```

- Note: Streamlit apps and FastAPI are different; pick one for Azure Web App deployment.

6) Environment variables
- If you want to enable poster downloads from TMDB, set the environment variable `TMDB_API_KEY` in Azure App Settings.

7) Debugging
- Check logs in Azure (Log stream) to see startup errors.
- Verify the presence of CSV files in the deployed app's filesystem.
