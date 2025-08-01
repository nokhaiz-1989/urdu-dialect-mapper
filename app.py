import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set page configuration
st.set_page_config(page_title="Urdu Dialect Mapper", layout="wide")

st.title("🗺️ Urdu Dialect Mapping and Profiling System")
st.markdown("Explore Urdu dialects across regions with audio, linguistic features, and similarity analysis.")

# Load CSV
csv_file = "dialect_samples_extended.csv"

required_columns = {
    "Example Phrase", "Dialect Cluster", "Region", "Latitude", "Longitude",
    "Audio File", "Morphological Tag", "Semantic Feature", 
    "Phonetic Variation", "Syntactic Structure"
}

if not os.path.isfile(csv_file):
    st.error(f"❌ File not found: {csv_file}")
    st.warning(f"❗ CSV must contain these columns: {required_columns}")
    st.stop()

df = pd.read_csv(csv_file)

if not required_columns.issubset(df.columns):
    st.error("❌ CSV is missing one or more required columns.")
    st.warning(f"❗ Required columns: {required_columns}")
    st.stop()

# Show map
st.subheader("🧭 Dialect Distribution Map")

m = folium.Map(location=[30.3753, 69.3451], zoom_start=5.2)

for _, row in df.iterrows():
    popup_text = f"""
    <b>Dialect:</b> {row['Dialect Cluster']}<br>
    <b>Phrase:</b> {row['Example Phrase']}<br>
    <b>Region:</b> {row['Region']}<br>
    <b>Morphology:</b> {row['Morphological Tag']}<br>
    <b>Phonetics:</b> {row['Phonetic Variation']}<br>
    <b>Semantics:</b> {row['Semantic Feature']}<br>
    <b>Syntax:</b> {row['Syntactic Structure']}<br>
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text,
        tooltip=row['Dialect Cluster'],
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

st_data = st_folium(m, width=700, height=500)

# Dialect details
st.subheader("🔎 Explore Phrases by Dialect")

dialect_options = sorted(df['Dialect Cluster'].unique())
selected_dialect = st.selectbox("Choose a dialect cluster", dialect_options)

filtered_df = df[df['Dialect Cluster'] == selected_dialect]

for _, row in filtered_df.iterrows():
    st.markdown(f"**🗣️ Phrase:** {row['Example Phrase']}")
    st.markdown(f"- 📍 Region: {row['Region']}")
    st.markdown(f"- 🧬 Morphological Tag: {row['Morphological Tag']}")
    st.markdown(f"- 🗣️ Phonetic Variation: {row['Phonetic Variation']}")
    st.markdown(f"- 🧠 Semantic Feature: {row['Semantic Feature']}")
    st.markdown(f"- 🧵 Syntactic Structure: {row['Syntactic Structure']}")

    # Audio playback
    audio_path = row['Audio File']
    if os.path.isfile(audio_path):
        ext = os.path.splitext(audio_path)[1].lower()
        audio_format = "audio/mp3"  # default
        if ext == ".m4a":
            audio_format = "audio/m4a"
        elif ext == ".wav":
            audio_format = "audio/wav"
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format=audio_format)
    else:
        st.info(f"🔇 Audio file not found: `{audio_path}`")

    st.markdown("---")

# Dialect similarity
st.subheader("📊 Dialect Similarity Matrix")

phrases = df['Example Phrase'].astype(str)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(phrases)
similarity_matrix = cosine_similarity(tfidf_matrix)

# Add index as dialect + region
df_index = df['Dialect Cluster'] + " | " + df['Region']
similarity_df = pd.DataFrame(similarity_matrix, index=df_index, columns=df_index)

st.dataframe(similarity_df.style.background_gradient(cmap='Blues'), use_container_width=True)

st.success("✅ App loaded successfully. Explore dialectal diversity now!")

