# World Development Cluster Analysis — Improved Streamlit Dashboard

## Files
- `app.py` — interactive dashboard
- `model_evaluation_cell.py` — improved evaluation cell to paste into the notebook after data preparation
- `World_development_mesurement.xlsx` — supplied dataset
- `requirements.txt` — dependencies

## Run locally (Windows / VS Code)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Keep `app.py`, `requirements.txt` and the Excel file in the same folder.

## Deploy
1. Upload `app.py`, `requirements.txt`, and `World_development_mesurement.xlsx` to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Create app, select your repository/branch, set main file to `app.py`, then deploy.

## Improvements in this version
- K selection comparison: Elbow, Silhouette, Davies-Bouldin and Calinski-Harabasz.
- Four-model comparison with DBSCAN noise coverage explicitly reported.
- PCA scatter plot, cluster count and cluster share charts.
- Country search, cluster filter and selected-country average profile.
- Cluster means, standardized profile heatmap and cautious automatic descriptions.
- Indicator comparison chart and downloadable cluster assignments.
- Upload another Excel/CSV file with the same 10 required feature columns.

## Model selection note
For the supplied data and notebook settings (K=3, StandardScaler, median imputation), K-Means and Agglomerative are nearly tied on Silhouette. Agglomerative has a slightly higher Silhouette and lower Davies-Bouldin; K-Means has a higher Calinski-Harabasz score. Therefore, justify K-Means as a practical, interpretable final model, not as the winner on every metric.

## Important
The supplied dataset contains multiple records for many countries (2,704 rows and 208 unique country names). The country explorer therefore shows the average of that country's records and the underlying source rows; it does not assume there is only one row per country.
