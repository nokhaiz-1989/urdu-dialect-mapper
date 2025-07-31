import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# Set up the page
st.set_page_config(page_title="Urdu Dialect Mapping Tool", layout="wide")

st.title("📍 Digital Dialectal Mapping and Profiling System for Urdu")
st.markdown("""
Explore **Urdu dialect clusters** across regions of Pakistan with enhanced linguistic profiling features.
""")

# Load data
@st.cache_data

def load_data():
    try:
        df = pd.read_csv("dialect_samples_with_locations.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File not found. Ensure 'dialect_samples_with_locations.csv' is in the correct folder.")
        return pd.DataFrame()

df = load_data()

# Required columns
required_cols = {"Example Phrase", "Dialect Cluster", "Region", "Latitude", "Longitude"}
if not required_cols.issubset(df.columns):
    st.error(f"❗ CSV file must contain these columns: {required_cols}")
    st.stop()

# Sidebar filters
st.sidebar.header("🎯 Filter Dialect Clusters")
clusters = df["Dialect Cluster"].unique().tolist()
selected_clusters = st.sidebar.multiselect("Select Cluster(s):", options=clusters, default=clusters)
filtered_df = df[df["Dialect Cluster"].isin(selected_clusters)]

# Phrase search
search_term = st.sidebar.text_input("🔍 Search Phrase:")
if search_term:
    filtered_df = filtered_df[filtered_df["Example Phrase"].str.contains(search_term, case=False, na=False)]

# Show filtered data
with st.expander("📋 View Filtered Dialect Data"):
    st.dataframe(filtered_df, use_container_width=True)

# Pie chart
st.subheader("📊 Dialect Distribution")
pie = px.pie(
    filtered_df,
    names="Dialect Cluster",
    title="Distribution by Dialect Cluster",
    hole=0.4
)
st.plotly_chart(pie, use_container_width=True)

# Dialect map
st.subheader("🗺️ Geographical Mapping of Urdu Dialects")
map_fig = px.scatter_mapbox(
    filtered_df,
    lat="Latitude",
    lon="Longitude",
    color="Dialect Cluster",
    hover_name="Region",
    hover_data=["Example Phrase"],
    zoom=5,
    height=600
)
map_fig.update_layout(mapbox_style="open-street-map")
map_fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
st.plotly_chart(map_fig, use_container_width=True)

# Optional: Audio playback if Audio File column exists
if "Audio File" in df.columns:
    st.subheader("🔊 Audio Playback for Sample Phrases")
    for i, row in filtered_df.iterrows():
        st.markdown(f"**{row['Dialect Cluster']} | {row['Region']}**: {row['Example Phrase']}")
        st.audio(row["Audio File"], format="audio/mp3")

# Dialect Similarity Analysis
st.subheader("📈 Dialect Phrase Similarity Analysis")
if len(filtered_df) >= 2:
    phrases = filtered_df["Example Phrase"].tolist()
    vectorizer = TfidfVectorizer().fit_transform(phrases)
    similarity_matrix = cosine_similarity(vectorizer)
    st.write("**Cosine Similarity Matrix:**")
    st.dataframe(pd.DataFrame(similarity_matrix, columns=phrases, index=phrases))

    st.write("**Levenshtein Distance (Pairwise):**")
    dist_matrix = [[Levenshtein.distance(p1, p2) for p2 in phrases] for p1 in phrases]
    st.dataframe(pd.DataFrame(dist_matrix, columns=phrases, index=phrases))
else:
    st.info("Please select at least two dialect samples to view similarity analysis.")

# Linguistic Annotations (if present)
st.subheader("✍️ Linguistic Annotations")
if set(["Morphological Tag", "Semantic Feature", "Phonetic Variation", "Syntactic Structure"]).issubset(df.columns):
    st.dataframe(filtered_df[["Example Phrase", "Morphological Tag", "Semantic Feature", "Phonetic Variation", "Syntactic Structure"]])
else:
    st.warning("⚠️ Linguistic annotation columns are missing. Please add them in the CSV if available.")

# Footer
st.markdown("---")
st.markdown("📌 Developed to assist linguistic researchers in **phonetic profiling**, **dialect clustering**, and **speaker variation** studies across Pakistan.")
