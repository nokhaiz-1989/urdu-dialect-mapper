import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import difflib
import os

# Set up the page
st.set_page_config(page_title="Urdu Dialect Mapping Tool", layout="wide")

# Title and description
st.title("📍 Digital Dialectal Mapping and Profiling System for Urdu")
st.markdown("""
This tool visualizes **Urdu dialect clusters** across regions of Pakistan.
You can explore regional phrases, listen to voice samples, and analyze linguistic patterns.
""")

# Load data
@st.cache_data
def load_data():
    file_path = "dialect_samples_extended.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    return df

df = load_data()

required_cols = {"Example Phrase", "Dialect Cluster", "Region", "Latitude", "Longitude", 
                 "Audio File", "Morphological Tag", "Semantic Feature", 
                 "Phonetic Variation", "Syntactic Structure"}
if not required_cols.issubset(df.columns):
    st.error(f"❗ CSV must contain these columns:\n{required_cols}")
    st.stop()

# Sidebar: Filter clusters
st.sidebar.header("🎯 Filter Dialect Clusters")
clusters = df["Dialect Cluster"].unique().tolist()
selected_clusters = st.sidebar.multiselect("Select Cluster(s):", options=clusters, default=clusters)

filtered_df = df[df["Dialect Cluster"].isin(selected_clusters)]

# Show data
st.subheader("📋 Dialect Samples with Linguistic Annotations")
st.dataframe(filtered_df.drop(columns=["Audio File"]), use_container_width=True)

# Play audio samples
st.subheader("🔊 Play Audio Samples")
for idx, row in filtered_df.iterrows():
    st.markdown(f"**{row['Dialect Cluster']} – {row['Region']}**")
    st.text(f"Phrase: {row['Example Phrase']}")
    if os.path.exists(row['Audio File']):
        st.audio(row['Audio File'], format='audio/mp3')
    else:
        st.warning("Audio sample not available.")

# Pie chart
st.subheader("📊 Dialect Distribution")
pie = px.pie(
    filtered_df,
    names="Dialect Cluster",
    title="Distribution by Dialect Cluster",
    hole=0.4
)
st.plotly_chart(pie, use_container_width=True)

# Map
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

# Similarity analysis
st.subheader("📈 Dialect Similarity (Levenshtein Distance)")
phrase_options = filtered_df["Example Phrase"].tolist()
phrase1 = st.selectbox("Select Phrase 1", phrase_options, key="p1")
phrase2 = st.selectbox("Select Phrase 2", phrase_options, key="p2")

def levenshtein_ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

similarity = levenshtein_ratio(phrase1, phrase2)
st.info(f"Similarity between selected phrases: **{similarity:.2f}**")

# Footer
st.markdown("---")
st.markdown("📌 Developed for linguistic profiling and dialect recognition in Pakistan.")
