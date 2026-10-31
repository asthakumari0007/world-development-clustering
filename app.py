import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="World Development Cluster Analysis", page_icon="🌍", layout="wide")

FEATURES = [
    "Birth Rate", "CO2 Emissions", "Energy Usage", "Health Exp % GDP",
    "Infant Mortality Rate", "Internet Usage", "Life Expectancy Female",
    "Life Expectancy Male", "Mobile Phone Usage", "Population Urban"
]
DEFAULT_FILE = "World_development_mesurement.xlsx"

@st.cache_data(show_spinner=False)
def load_data(file_bytes, file_name):
    from io import BytesIO
    bio = BytesIO(file_bytes)
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(bio)
    return pd.read_excel(bio)

@st.cache_data(show_spinner=False)
def prepare_data(raw):
    missing = [c for c in FEATURES if c not in raw.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    numeric = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid = ~numeric.isna().all(axis=1)
    clean_data = raw.loc[valid].copy().reset_index(drop=True)
    X = numeric.loc[valid].copy().reset_index(drop=True)
    medians = X.median()
    X = X.fillna(medians)
    no_info = X.columns[X.isna().all()].tolist()
    if no_info:
        raise ValueError("These selected indicators have no usable values: " + ", ".join(no_info))
    return clean_data, X

@st.cache_data(show_spinner=False)
def evaluate_k_values(X, max_k=10):
    scaled = StandardScaler().fit_transform(X)
    rows = []
    for k in range(2, min(max_k, len(X)-1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score": silhouette_score(scaled, labels),
            "Davies-Bouldin Index": davies_bouldin_score(scaled, labels),
            "Calinski-Harabasz Score": calinski_harabasz_score(scaled, labels)
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def fit_final_kmeans(X, k):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    return scaled, labels, coords, float(pca.explained_variance_ratio_.sum())

@st.cache_data(show_spinner=False)
def compare_models(X, k):
    scaled = StandardScaler().fit_transform(X)
    models = {
        "K-Means": KMeans(n_clusters=k, random_state=42, n_init=20),
        "Agglomerative": AgglomerativeClustering(n_clusters=k),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=8),
        "Gaussian Mixture": GaussianMixture(n_components=k, random_state=42, n_init=5)
    }
    rows = []
    for name, model in models.items():
        labels = model.fit_predict(scaled)
        noise_mask = labels != -1 if name == "DBSCAN" else np.ones(len(labels), dtype=bool)
        scored_labels, scored_data = labels[noise_mask], scaled[noise_mask]
        n_clusters = len(np.unique(scored_labels))
        valid_metric = n_clusters >= 2 and len(scored_labels) > n_clusters
        rows.append({
            "Algorithm": name,
            "Clusters (excluding noise)": n_clusters,
            "Noise records": int((labels == -1).sum()) if name == "DBSCAN" else 0,
            "Records scored": int(noise_mask.sum()),
            "Silhouette Score": silhouette_score(scored_data, scored_labels) if valid_metric else np.nan,
            "Davies-Bouldin Index": davies_bouldin_score(scored_data, scored_labels) if valid_metric else np.nan,
            "Calinski-Harabasz Score": calinski_harabasz_score(scored_data, scored_labels) if valid_metric else np.nan
        })
    return pd.DataFrame(rows)

def profile_interpretation(profile):
    """Create cautious, data-driven summaries from standardized cluster means."""
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
    descriptions = {}
    higher_good = {"Internet Usage", "Life Expectancy Female", "Life Expectancy Male",
                   "Mobile Phone Usage", "Population Urban", "Health Exp % GDP"}
    lower_good = {"Birth Rate", "Infant Mortality Rate"}
    for cluster in profile.index:
        row = z.loc[cluster].dropna()
        high = row.sort_values(ascending=False).head(3).index.tolist()
        low = row.sort_values().head(2).index.tolist()
        notes = []
        if any(f in higher_good for f in high):
            notes.append("relatively higher " + ", ".join([f for f in high if f in higher_good][:2]))
        if any(f in lower_good for f in low):
            notes.append("relatively lower " + ", ".join([f for f in low if f in lower_good][:2]))
        if "CO2 Emissions" in high or "Energy Usage" in high:
            notes.append("high energy use / emissions indicators")
        descriptions[int(cluster)] = "; ".join(notes) if notes else "mixed indicator pattern"
    return descriptions, z

st.title("🌍 World Development Cluster Analysis")
st.write("An interactive analysis of country development indicators using unsupervised machine learning.")
st.caption("The application follows the project notebook: 10 numeric indicators, median imputation, StandardScaler and K-Means clustering.")

with st.sidebar:
    st.header("Dashboard settings")
    upload = st.file_uploader("Upload Excel or CSV dataset", type=["xlsx", "xls", "csv"])
    st.divider()
    k = st.slider("Number of clusters (K)", 2, 10, 3)
    st.caption("K = 3 is the notebook's selected setting. You can test other K values.")
    st.caption("Clustering uses all 10 indicators. PCA is only used to display the clusters in 2D.")

try:
    if upload is not None:
        raw = load_data(upload.getvalue(), upload.name)
    else:
        raw = load_data(open(DEFAULT_FILE, "rb").read(), DEFAULT_FILE)
    data, X = prepare_data(raw)
    scaled, labels, coords, pca_var = fit_final_kmeans(X, k)
except FileNotFoundError:
    st.error(f"Dataset file '{DEFAULT_FILE}' was not found. Keep it in the same folder as app.py or upload it from the sidebar.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to process the dataset: {exc}")
    st.stop()

country_col = "Country" if "Country" in data.columns else None
result = data.copy()
result["Cluster"] = labels.astype(int)
result["PCA 1"] = coords[:, 0]
result["PCA 2"] = coords[:, 1]
counts = result["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Records")
profile = X.assign(Cluster=labels).groupby("Cluster").mean()
profile_display = profile.round(3)
interpretations, profile_z = profile_interpretation(profile)

# Key metrics
st.subheader("Project overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records analysed", f"{len(result):,}")
c2.metric("Countries in dataset", f"{data[country_col].nunique():,}" if country_col else "Not available")
c3.metric("Indicators used", len(FEATURES))
c4.metric("PCA variance shown", f"{pca_var*100:.1f}%")

overview, country_tab, cluster_tab, evaluation_tab, data_tab, compare_tab, insights_tab = st.tabs([
    "📊 Overview", "🌐 Country explorer", "🧩 Cluster interpretation",
    "📈 Model evaluation", "📁 Data & downloads",
    "🔄 Country comparison", "💡 Smart insights"
])

with overview:
    left, right = st.columns(2)
    with left:
        fig = px.bar(counts, x="Cluster", y="Records", color="Cluster", text="Records",
                     title="Records in each K-Means cluster")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.pie(counts, names="Cluster", values="Records", hole=.45,
                     title="Share of records by cluster")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cluster distribution summary")
    distribution = counts.copy()
    distribution["Percentage"] = (distribution["Records"] / len(result) * 100).round(2)
    st.dataframe(distribution, use_container_width=True, hide_index=True)
    fig = px.bar(distribution, x="Cluster", y="Percentage", text="Percentage",
                 title="Percentage of records in each cluster", color="Cluster")
    fig.update_traces(texttemplate="%{text:.2f}%")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(result, x="PCA 1", y="PCA 2", color=result["Cluster"].astype(str),
                     hover_name=country_col if country_col else None,
                     hover_data={"Cluster": True, "PCA 1": False, "PCA 2": False},
                     title="K-Means clusters in two-dimensional PCA space",
                     labels={"color": "Cluster"})
    fig.update_traces(marker={"size": 8, "opacity": .72})
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"The first two PCA components explain {pca_var*100:.1f}% of the variance. This chart is a 2D view; the K-Means model was trained on all 10 scaled indicators.")

with country_tab:
    st.subheader("Find a country and inspect its cluster")
    if not country_col:
        st.warning("The dataset has no 'Country' column, so country search is unavailable.")
    else:
        countries = sorted(data[country_col].dropna().astype(str).unique())
        q = st.text_input("Search country", placeholder="Start typing a country name")
        matching = [name for name in countries if q.lower() in name.lower()] if q else countries
        cluster_filter = st.selectbox("Cluster filter", ["All"] + [int(x) for x in sorted(result["Cluster"].unique())])
        filtered = result[result[country_col].astype(str).isin(matching)].copy()
        if cluster_filter != "All":
            filtered = filtered[filtered["Cluster"] == cluster_filter]
        st.write(f"**{len(filtered):,} records** match your search and filter.")
        if not filtered.empty:
            selected_country = st.selectbox("Select country for its summary", sorted(filtered[country_col].dropna().astype(str).unique()))
            country_rows = filtered[filtered[country_col].astype(str) == selected_country]
            # If source has a year/time field, retain it in the time series; otherwise show mean and source records.
            st.markdown(f"#### {selected_country}")
            st.metric("Most frequent assigned cluster", int(country_rows["Cluster"].mode().iloc[0]))
            summary = country_rows[FEATURES].mean().to_frame("Average indicator value").round(4)
            st.dataframe(summary, use_container_width=True)
            st.markdown("**Source records for this country**")
            st.dataframe(country_rows.drop(columns=["PCA 1", "PCA 2"], errors="ignore"), use_container_width=True, hide_index=True)
        else:
            st.warning("No countries match the current search/filter.")

with cluster_tab:
    st.subheader("What characterizes each cluster?")
    st.caption("These are descriptive patterns in the selected indicators, not official development rankings or causal conclusions.")
    st.dataframe(counts, use_container_width=True, hide_index=True)
    st.markdown("#### Average indicator values")
    st.dataframe(profile_display, use_container_width=True)
    st.markdown("#### Relative cluster patterns (standardized means)")
    heat = px.imshow(profile_z, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                     origin="lower", title="Indicator levels relative to the overall average")
    st.plotly_chart(heat, use_container_width=True)
    for cluster_id, description in interpretations.items():
        st.markdown(f"**Cluster {cluster_id}:** {description}.")
    indicator = st.selectbox("Compare one indicator across clusters", FEATURES,
                             index=FEATURES.index("Life Expectancy Female"))
    box = px.box(result, x=result["Cluster"].astype(str), y=indicator, color=result["Cluster"].astype(str),
                 points="outliers", title=f"{indicator} by cluster",
                 labels={"x": "Cluster", "y": indicator, "color": "Cluster"})
    st.plotly_chart(box, use_container_width=True)

with evaluation_tab:
    st.subheader("1. Choose K using multiple metrics")
    st.write("Compare K values rather than relying on only one score. Higher Silhouette and Calinski-Harabasz are generally preferred; lower Davies-Bouldin is generally preferred.")
    k_eval = evaluate_k_values(X, 10)
    best_sil_k = int(k_eval.loc[k_eval["Silhouette Score"].idxmax(), "K"])
    st.write(f"Best tested K by Silhouette Score: **{best_sil_k}**. Current dashboard K: **{k}**.")
    e1, e2 = st.columns(2)
    with e1:
        fig = px.line(k_eval, x="K", y="Inertia", markers=True, title="Elbow method")
        st.plotly_chart(fig, use_container_width=True)
    with e2:
        fig = px.line(k_eval, x="K", y="Silhouette Score", markers=True, title="Silhouette score by K")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(k_eval.round(4), use_container_width=True, hide_index=True)

    st.subheader("2. Compare four clustering algorithms")
    if st.button("Run model comparison for current K", type="primary"):
        with st.spinner("Evaluating K-Means, Agglomerative, DBSCAN and Gaussian Mixture..."):
            comp = compare_models(X, k)
        st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
        st.caption("DBSCAN metrics are calculated only on non-noise records. Always inspect its noise count and records scored before comparing its scores with other models.")
        st.markdown("**How to read the metrics:** Silhouette = higher is better; Davies-Bouldin = lower is better; Calinski-Harabasz = higher is generally better.")
    else:
        st.info("Click the button to run the four-model comparison. This may take a little time.")

    st.subheader("3. Current K-Means result")
    current_sil = silhouette_score(scaled, labels)
    current_dbi = davies_bouldin_score(scaled, labels)
    current_ch = calinski_harabasz_score(scaled, labels)
    a, b, c = st.columns(3)
    a.metric("Silhouette", f"{current_sil:.4f}")
    b.metric("Davies-Bouldin", f"{current_dbi:.4f}")
    c.metric("Calinski-Harabasz", f"{current_ch:.1f}")
    st.caption("Scores change when you change K or upload a different dataset.")

with data_tab:
    st.subheader("Cleaned dataset with cluster assignments")
    st.write(f"Original rows: {len(raw):,} | Records used: {len(data):,} | Rows with all 10 indicators missing were removed.")
    st.dataframe(result.drop(columns=["PCA 1", "PCA 2"], errors="ignore"), use_container_width=True, hide_index=True)
    csv = result.drop(columns=["PCA 1", "PCA 2"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download all country/record cluster assignments", csv,
                       "world_development_kmeans_results.csv", "text/csv")



with compare_tab:
    st.subheader("Compare development indicators between two countries")
    if not country_col:
        st.warning("The dataset has no 'Country' column, so country comparison is unavailable.")
    else:
        country_names = sorted(data[country_col].dropna().astype(str).unique())
        if len(country_names) < 2:
            st.warning("At least two countries are required for comparison.")
        else:
            left, right = st.columns(2)
            country_a = left.selectbox("First country", country_names, index=0, key="compare_country_a")
            country_b = right.selectbox("Second country", country_names, index=1, key="compare_country_b")
            compare_features = st.multiselect(
                "Select indicators to compare",
                FEATURES,
                default=["Life Expectancy Female", "Internet Usage", "Birth Rate", "Infant Mortality Rate"],
                key="compare_features"
            )
            if country_a == country_b:
                st.info("Please select two different countries.")
            elif not compare_features:
                st.info("Select at least one indicator.")
            else:
                country_means = X.assign(Country=data[country_col].astype(str)).groupby("Country")[compare_features].mean()
                comparison = country_means.loc[[country_a, country_b]].T
                comparison.columns = [country_a, country_b]
                st.dataframe(comparison.round(3), use_container_width=True)
                plot_data = comparison.reset_index().melt(
                    id_vars="index", var_name="Country", value_name="Average value"
                ).rename(columns={"index": "Indicator"})
                fig = px.bar(
                    plot_data, x="Indicator", y="Average value", color="Country",
                    barmode="group", title="Selected development indicators by country"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Values are averages across available records for each country. Indicators use their original units and scales.")

with insights_tab:
    st.subheader("Data-driven development insights")
    st.caption("Insights below are calculated from the current dataset and current K-Means cluster assignments. They describe patterns; they do not prove causation or rank countries.")
    selected_indicator = st.selectbox(
        "Choose an indicator for cluster insights",
        FEATURES,
        index=FEATURES.index("Life Expectancy Female"),
        key="insight_indicator"
    )
    insight_profile = X.assign(Cluster=labels).groupby("Cluster")[selected_indicator].mean().sort_values(ascending=False)
    high_cluster = int(insight_profile.index[0])
    low_cluster = int(insight_profile.index[-1])
    high_value = float(insight_profile.iloc[0])
    low_value = float(insight_profile.iloc[-1])
    m1, m2, m3 = st.columns(3)
    m1.metric("Highest cluster average", f"Cluster {high_cluster}")
    m2.metric("Highest average value", f"{high_value:,.3f}")
    m3.metric("Lowest average value", f"{low_value:,.3f}")
    if high_cluster != low_cluster:
        st.write(
            f"For **{selected_indicator}**, Cluster {high_cluster} has the highest average "
            f"({high_value:,.3f}), while Cluster {low_cluster} has the lowest average "
            f"({low_value:,.3f})."
        )
    else:
        st.write(
            f"All records are assigned to Cluster {high_cluster} for this current K setting."
        )

    insight_table = pd.DataFrame({
        "Cluster": insight_profile.index.astype(int),
        "Average indicator value": insight_profile.values.astype(float)
    })

    insight_table["Average indicator value"] = (
        insight_table["Average indicator value"].round(3)
    )

    st.dataframe(
        insight_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cluster": st.column_config.NumberColumn("Cluster"),
            "Average indicator value": st.column_config.NumberColumn(
                "Average indicator value",
                format="%.3f"
            )
        }
    )

    fig = px.bar(
        insight_table,
        x="Cluster",
        y="Average indicator value",
        color="Cluster",
        title=f"Average {selected_indicator} by cluster"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("Interpret each indicator in its own units.")
    st.divider()

    st.caption(
        "World Development Cluster Analysis | K-Means | StandardScaler | PCA | Streamlit"
    )