# MODEL EVALUATION — run after X, X_scaled and k_results_df are prepared
# This cell compares the four algorithms using three clustering metrics.
# Higher Silhouette / Calinski-Harabasz is generally better.
# Lower Davies-Bouldin is generally better.

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Use the same cleaned feature matrix X from the notebook.
X_scaled = StandardScaler().fit_transform(X)
K = 3  # selected K from the K-value evaluation

models = {
    "K-Means": KMeans(n_clusters=K, random_state=42, n_init=20),
    "Agglomerative": AgglomerativeClustering(n_clusters=K),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=8),
    "Gaussian Mixture": GaussianMixture(n_components=K, random_state=42, n_init=5)
}

comparison_rows = []
model_labels = {}

for model_name, model in models.items():
    labels = model.fit_predict(X_scaled)
    model_labels[model_name] = labels

    # DBSCAN assigns -1 to noise. Report how many records are excluded.
    valid_mask = labels != -1 if model_name == "DBSCAN" else np.ones(len(labels), dtype=bool)
    scored_labels = labels[valid_mask]
    scored_data = X_scaled[valid_mask]
    cluster_count = len(np.unique(scored_labels))
    valid_score = cluster_count >= 2 and len(scored_labels) > cluster_count

    comparison_rows.append({
        "Model": model_name,
        "Clusters": cluster_count,
        "Noise Records": int((labels == -1).sum()) if model_name == "DBSCAN" else 0,
        "Records Scored": int(valid_mask.sum()),
        "Silhouette Score": silhouette_score(scored_data, scored_labels) if valid_score else np.nan,
        "Davies-Bouldin Index": davies_bouldin_score(scored_data, scored_labels) if valid_score else np.nan,
        "Calinski-Harabasz Score": calinski_harabasz_score(scored_data, scored_labels) if valid_score else np.nan
    })

model_comparison = pd.DataFrame(comparison_rows)
display(model_comparison.round(4))

# Recommended interpretation for this dataset:
# K-Means and Agglomerative are very close on Silhouette.
# Do not claim K-Means has the highest Silhouette if Agglomerative is higher.
# Choose based on all metrics, interpretability, cluster sizes and project requirements.
